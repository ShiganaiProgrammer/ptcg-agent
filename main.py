import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

MAX_OPTIONS = 64
MAX_BENCH = 5
OBS_DIM = 61
HIDDEN_DIM = 256

CARD_TYPE_POKEMON = 0
CARD_TYPE_ITEM = 1
CARD_TYPE_TOOL = 2
CARD_TYPE_SUPPORTER = 3
CARD_TYPE_STADIUM = 4
CARD_TYPE_ENERGY = 5

# ============================================================
# Deck loading
# ============================================================
_DECK = None

def _load_deck():
    global _DECK
    if _DECK is not None:
        return _DECK
    paths = ["deck.csv", "/kaggle_simulations/agent/deck.csv"]
    try:
        paths.insert(1, os.path.join(os.path.dirname(__file__), "deck.csv"))
    except NameError:
        pass
    for path in paths:
        if os.path.exists(path):
            with open(path) as f:
                lines = f.read().strip().split("\n")
            _DECK = [int(x) for x in lines if x.strip()][:60]
            return _DECK
    _DECK = []
    return _DECK

# ============================================================
# Compat layer: dict-style access for dicts AND objects
# ============================================================
class _DictCompat:
    __slots__ = ('_obj',)
    def __init__(self, obj):
        self._obj = obj
    def get(self, key, default=None):
        if isinstance(self._obj, dict):
            val = self._obj.get(key, default)
        else:
            val = getattr(self._obj, key, default)
        return _wrap(val)
    def __getitem__(self, key):
        if isinstance(self._obj, dict):
            return _wrap(self._obj[key])
        return _wrap(getattr(self._obj, key))
    def __bool__(self):
        return self._obj is not None
    def __contains__(self, key):
        if isinstance(self._obj, dict):
            return key in self._obj
        return hasattr(self._obj, key)

def _wrap(obj):
    if obj is None or isinstance(obj, (bool, int, float, str, bytes, np.ndarray)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_wrap(x) for x in obj]
    return _DictCompat(obj)

# ============================================================
# Neural Network
# ============================================================
class PolicyNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, action_dim=MAX_OPTIONS, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action_mask=None):
        h = self.shared(obs)
        logits = self.policy(h)
        if action_mask is not None:
            mask = action_mask.float()
            logits = logits * mask - 100.0 * (1.0 - mask)
        probs = F.softmax(logits, dim=-1)
        value = self.value(h)
        return probs, value

# ============================================================
# Observation Encoder (works with _DictCompat wrapped objects)
# ============================================================
def encode_observation(obs_dict) -> np.ndarray:
    current = obs_dict.get("current")
    features = []

    your_idx = current.get("yourIndex", 0) if current else 0
    opponent_idx = 1 - your_idx
    players = current.get("players", [None, None]) if current else [None, None]
    me = _wrap(players[your_idx]) if players[your_idx] is not None else _wrap({})
    opp = _wrap(players[opponent_idx]) if players[opponent_idx] is not None else _wrap({})

    if current:
        features.append(current.get("turn", 0) / 50.0)
        features.append(float(current.get("firstPlayer", -1)))
        features.append(float(current.get("supporterPlayed", False)))
        features.append(float(current.get("stadiumPlayed", False)))
        features.append(float(current.get("energyAttached", False)))
        features.append(float(current.get("retreated", False)))
        features.append(current.get("result", -1) + 1)
    else:
        features.extend([0.0] * 7)

    active_list = me.get("active", [])
    if active_list and active_list[0] is not None:
        poke = _wrap(active_list[0])
        features.append(poke.get("hp", 0) / 200.0)
        features.append(poke.get("maxHp", 0) / 200.0)
        features.append(len(poke.get("energies", [])) / 10.0)
        features.append(float(poke.get("appearThisTurn", False)))
        features.append(1.0 if poke.get("preEvolution") else 0.0)
    else:
        features.extend([0.0] * 5)

    bench = me.get("bench", [])
    for i in range(MAX_BENCH):
        if i < len(bench) and bench[i] is not None:
            poke = _wrap(bench[i])
            features.append(poke.get("hp", 0) / 200.0)
            features.append(len(poke.get("energies", [])) / 10.0)
            features.append(1.0 if poke.get("preEvolution") else 0.0)
        else:
            features.extend([0.0] * 3)
    features.append(len(bench) / 5.0)

    hand = me.get("hand", [])
    features.append(min(len(hand), 10) / 10.0)
    if hand:
        pokemon_in_hand = sum(1 for c in hand if _wrap(c).get("cardType") == CARD_TYPE_POKEMON) / 10.0
        trainer_in_hand = sum(1 for c in hand if _wrap(c).get("cardType") not in (CARD_TYPE_POKEMON, CARD_TYPE_ENERGY)) / 10.0
        energy_in_hand = sum(1 for c in hand if _wrap(c).get("cardType") == CARD_TYPE_ENERGY) / 10.0
        features.extend([pokemon_in_hand, trainer_in_hand, energy_in_hand])
    else:
        features.extend([0.0, 0.0, 0.0])

    features.append(me.get("deckCount", 0) / 60.0)

    prize = me.get("prize", [])
    prize_remaining = sum(1 for p in prize if p is not None)
    features.append(prize_remaining / 6.0)

    features.extend([
        float(me.get("poisoned", False)),
        float(me.get("burned", False)),
        float(me.get("asleep", False)),
        float(me.get("paralyzed", False)),
        float(me.get("confused", False)),
    ])

    opp_active_list = opp.get("active", [])
    if opp_active_list and opp_active_list[0] is not None:
        poke = _wrap(opp_active_list[0])
        features.append(poke.get("hp", 0) / 200.0)
        features.append(len(poke.get("energies", [])) / 10.0)
        features.append(1.0 if poke.get("preEvolution") else 0.0)
    else:
        features.extend([0.0] * 3)

    opp_bench = opp.get("bench", [])
    for i in range(MAX_BENCH):
        if i < len(opp_bench) and opp_bench[i] is not None:
            poke = _wrap(opp_bench[i])
            features.append(poke.get("hp", 0) / 200.0)
            features.append(len(poke.get("energies", [])) / 10.0)
        else:
            features.extend([0.0] * 2)
    features.append(len(opp_bench) / 5.0)

    features.append(opp.get("handCount", 0) / 10.0)
    features.append(opp.get("deckCount", 0) / 60.0)
    opp_prize = opp.get("prize", [])
    opp_prize_remaining = sum(1 for p in opp_prize if p is not None)
    features.append(opp_prize_remaining / 6.0)

    features.extend([
        float(opp.get("poisoned", False)),
        float(opp.get("burned", False)),
        float(opp.get("asleep", False)),
        float(opp.get("paralyzed", False)),
        float(opp.get("confused", False)),
    ])

    return np.array(features, dtype=np.float32)


def get_action_mask(obs_dict) -> np.ndarray:
    mask = np.zeros(MAX_OPTIONS, dtype=bool)
    select = obs_dict.get("select")
    if select:
        option = select.get("option")
        if option is not None:
            n_options = min(len(option), MAX_OPTIONS)
            mask[:n_options] = True
    return mask


# ============================================================
# Agent - Card Constants & Helpers
# ============================================================
_DEVICE = None
_POLICY = None

# Card IDs
MAKUHITA_ID = 673
HARIYAMA_ID = 674
LUNATONE_ID = 675
SOLROCK_ID = 676
RIOLU_ID = 677
MEGA_LUCARIO_ID = 678
LILLIES_DETERMINATION_ID = 1227

# Energy caps per Pokemon ID
ENERGY_CAPS = {
    MAKUHITA_ID: 3,
    HARIYAMA_ID: 3,
    RIOLU_ID: 2,
    MEGA_LUCARIO_ID: 2,
    SOLROCK_ID: 1,
    LUNATONE_ID: 2,
}

# Priority order for energy attachment (higher index = higher priority)
ENERGY_PRIORITY = [
    LUNATONE_ID,
    SOLROCK_ID,
    MAKUHITA_ID,
    HARIYAMA_ID,
    RIOLU_ID,
    MEGA_LUCARIO_ID,
]

DECKOUT_THRESHOLD = 5

# Ability tracking
_ABILITY_USED_THIS_TURN = False

# Fighting energy card ID (Basic Fighting Energy)
FIGHTING_ENERGY_ID = 241  # Basic Fighting Energy


def _get_my_raw_state(obs_dict):
    current = obs_dict.get("current")
    if current is None:
        return None
    players = current.get("players")
    if players is None:
        return None
    your_idx = current.get("yourIndex", 0)
    raw = players[your_idx] if your_idx < len(players) and players[your_idx] is not None else None
    if raw is None:
        return None
    if isinstance(raw, _DictCompat):
        return raw._obj
    return raw


def _get_active_poke(me_raw):
    if me_raw is None:
        return None
    active = me_raw.get("active") if isinstance(me_raw, dict) else getattr(me_raw, "active", None)
    if not active:
        return None
    first = active[0] if isinstance(active, (list, tuple)) else None
    if first is None:
        return None
    if isinstance(first, _DictCompat):
        return first._obj
    return first


def _get_field_pokemon(me_raw):
    if me_raw is None:
        return []
    result = []
    active = me_raw.get("active") if isinstance(me_raw, dict) else None
    if active and isinstance(active, list):
        for i, poke in enumerate(active):
            if poke is not None:
                p = poke._obj if isinstance(poke, _DictCompat) else poke
                result.append(("active", i, p))
    bench = me_raw.get("bench") if isinstance(me_raw, dict) else None
    if bench and isinstance(bench, list):
        for i, poke in enumerate(bench):
            if poke is not None:
                p = poke._obj if isinstance(poke, _DictCompat) else poke
                result.append(("bench", i, p))
    return result


def _get_field_pokemon_by_id(me_raw, card_id):
    for zone, idx, poke in _get_field_pokemon(me_raw):
        if poke.get("id") == card_id:
            return zone, idx, poke
    return None, None, None


def _is_card_in_hand(me_raw, card_id):
    if me_raw is None:
        return False
    hand = me_raw.get("hand") if isinstance(me_raw, dict) else None
    if not hand:
        return False
    for c in hand:
        if c is None:
            continue
        card = c._obj if isinstance(c, _DictCompat) else c
        if card.get("id") == card_id:
            return True
    return False


def _count_energy_in_hand(me_raw):
    if me_raw is None:
        return 0
    hand = me_raw.get("hand") if isinstance(me_raw, dict) else None
    if not hand:
        return 0
    count = 0
    for c in hand:
        if c is None:
            continue
        card = c._obj if isinstance(c, _DictCompat) else c
        if card.get("cardType") == CARD_TYPE_ENERGY:
            count += 1
    return count


def _find_ability_option(opts, area, index):
    for i, o in enumerate(opts):
        if isinstance(o, dict) and o.get("type") == 10:
            if o.get("area") == area and o.get("index") == index:
                return i
    return None


def _get_energy_priority_score(card_id):
    try:
        return ENERGY_PRIORITY.index(card_id)
    except ValueError:
        return -1


def _get_opp_raw_state(obs_dict):
    current = obs_dict.get("current")
    if current is None:
        return None
    players = current.get("players")
    if players is None:
        return None
    your_idx = current.get("yourIndex", 0)
    opp_idx = 1 - your_idx
    raw = players[opp_idx] if opp_idx < len(players) and players[opp_idx] is not None else None
    if raw is None:
        return None
    if isinstance(raw, _DictCompat):
        return raw._obj
    return raw


def _get_opp_field_pokemon(opp_raw):
    if opp_raw is None:
        return []
    result = []
    active = opp_raw.get("active") if isinstance(opp_raw, dict) else None
    if active and isinstance(active, list):
        for i, poke in enumerate(active):
            if poke is not None:
                p = poke._obj if isinstance(poke, _DictCompat) else poke
                result.append(("active", i, p))
    bench = opp_raw.get("bench") if isinstance(opp_raw, dict) else None
    if bench and isinstance(bench, list):
        for i, poke in enumerate(bench):
            if poke is not None:
                p = poke._obj if isinstance(poke, _DictCompat) else poke
                result.append(("bench", i, p))
    return result


def _get_policy():
    global _DEVICE, _POLICY
    if _POLICY is None:
        _DEVICE = torch.device("cpu")
        _POLICY = PolicyNet()
        model_paths = ["model.pt", "/kaggle_simulations/agent/model.pt"]
        try:
            model_paths.insert(0, os.path.join(os.path.dirname(__file__), "model.pt"))
        except NameError:
            pass
        for p in model_paths:
            if os.path.exists(p):
                _POLICY.load_state_dict(torch.load(p, map_location=_DEVICE, weights_only=True))
                break
        _POLICY.eval()
    return _POLICY


def _handle_ability_subselect(select, me_raw):
    opts = select.get("option", [])
    if not opts:
        return None

    sel_type = select.get("type")
    context = select.get("context")

    if sel_type == 1 and context == 8:
        for i, o in enumerate(opts):
            if isinstance(o, dict) and o.get("area") == 2:
                return [i]
        return [0]

    if sel_type == 1:
        for i, o in enumerate(opts):
            if isinstance(o, dict):
                return [i]
        return [0]

    return None


def agent(obs_dict) -> list[int]:
    global _ABILITY_USED_THIS_TURN

    obs_dict = _wrap(obs_dict)

    select = obs_dict.get("select")
    if select is None:
        _ABILITY_USED_THIS_TURN = False
        return _load_deck()

    num_to_select = select.get("maxCount", 1)

    opts = select.get("option", [])
    if not opts:
        return []

    me_raw = _get_my_raw_state(obs_dict)
    opp_raw = _get_opp_raw_state(obs_dict)

    # --- Ability sub-selection handling ---
    if _ABILITY_USED_THIS_TURN and select.get("type") == 1:
        sub_choice = _handle_ability_subselect(select, me_raw)
        if sub_choice is not None:
            _ABILITY_USED_THIS_TURN = False
            return sub_choice

    # Reset at start of main action
    if select.get("type") == 0:
        _ABILITY_USED_THIS_TURN = False

    # Rule: Deck-out protection
    if select.get("type") == 0:
        if me_raw is not None:
            deck_cnt = me_raw.get("deckCount", 60)
            if deck_cnt <= DECKOUT_THRESHOLD:
                for i, o in enumerate(opts):
                    if o.get("type") == 12:
                        return [i]

    # Rule: Ability - Lunar Cycle (Lunatone)
    if select.get("type") == 0 and not _ABILITY_USED_THIS_TURN:
        if me_raw is not None:
            lun_zone, lun_idx, lun_poke = _get_field_pokemon_by_id(me_raw, LUNATONE_ID)
            if lun_zone is not None and lun_poke is not None:
                sol_zone, sol_idx, sol_poke = _get_field_pokemon_by_id(me_raw, SOLROCK_ID)
                if sol_zone is not None:
                    riolu_zone, _, _ = _get_field_pokemon_by_id(me_raw, RIOLU_ID)
                    mega_zone, _, _ = _get_field_pokemon_by_id(me_raw, MEGA_LUCARIO_ID)
                    riolu_in_hand = _is_card_in_hand(me_raw, RIOLU_ID)
                    mega_in_hand = _is_card_in_hand(me_raw, MEGA_LUCARIO_ID)
                    no_lucario = (riolu_zone is None and mega_zone is None
                                  and not riolu_in_hand and not mega_in_hand)
                    if no_lucario:
                        energy_count = _count_energy_in_hand(me_raw)
                        if energy_count > 0:
                            lun_area = 4 if lun_zone == "active" else 5
                            opt_idx = _find_ability_option(opts, lun_area, lun_idx)
                            if opt_idx is not None:
                                _ABILITY_USED_THIS_TURN = True
                                return [opt_idx]

    # Rule: Ability - Heave-Ho Catcher (Hariyama)
    if select.get("type") == 0:
        if me_raw is not None and opp_raw is not None:
            hari_zone, hari_idx, hari_poke = _get_field_pokemon_by_id(me_raw, HARIYAMA_ID)
            if hari_zone is not None and hari_poke is not None:
                hari_damage = 210
                opp_bench = opp_raw.get("bench") if isinstance(opp_raw, dict) else []
                for bi, bp in enumerate(opp_bench):
                    if bp is None:
                        continue
                    poke = bp._obj if isinstance(bp, _DictCompat) else bp
                    if poke.get("hp", 0) <= hari_damage:
                        hari_area = 4 if hari_zone == "active" else 5
                        opt_idx = _find_ability_option(opts, hari_area, hari_idx)
                        if opt_idx is not None:
                            _ABILITY_USED_THIS_TURN = True
                            return [opt_idx]

    # Rule: Energy priority (attach only if active Pokemon is the best target)
    if select.get("type") == 0:
        if me_raw is not None:
            active = _get_active_poke(me_raw)
            if active is not None:
                pid = active.get("id")
                if pid is not None and pid in ENERGY_CAPS:
                    current_nrg = len(active.get("energies", []))
                    if current_nrg < ENERGY_CAPS[pid]:
                        for i, o in enumerate(opts):
                            if o.get("type") == 14:
                                return [i]

    # Rule: Play priority - Items > Lillie's Determination > other Supporters
    if select.get("type") == 0:
        if me_raw is not None:
            current = obs_dict.get("current")
            supporter_played = current.get("supporterPlayed", False) if current else False
            hand = me_raw.get("hand") if isinstance(me_raw, dict) else []
            if hand:
                item_opts = []
                lillie_opt = None
                supporter_opts = []
                for i, o in enumerate(opts):
                    if isinstance(o, dict) and o.get("type") == 8 and o.get("area") == 2:
                        idx = o.get("index")
                        if 0 <= idx < len(hand):
                            card = hand[idx]
                            card_obj = card._obj if isinstance(card, _DictCompat) else card
                            ctype = card_obj.get("cardType")
                            cid = card_obj.get("id")
                            if ctype in (CARD_TYPE_ITEM, CARD_TYPE_TOOL):
                                item_opts.append(i)
                            elif cid == LILLIES_DETERMINATION_ID and not supporter_played:
                                lillie_opt = i
                            elif ctype == CARD_TYPE_SUPPORTER and not supporter_played:
                                supporter_opts.append(i)
                if item_opts:
                    return [item_opts[0]]
                if lillie_opt is not None:
                    return [lillie_opt]
                if supporter_opts:
                    return [supporter_opts[0]]

    valid = np.where(get_action_mask(obs_dict))[0]
    if len(valid) == 0:
        return []

    policy = _get_policy()
    obs_t = torch.from_numpy(encode_observation(obs_dict)).float().unsqueeze(0)

    actions = []
    remaining = valid.tolist()

    with torch.no_grad():
        for _ in range(num_to_select):
            if not remaining:
                break
            current_mask = np.zeros(MAX_OPTIONS, dtype=bool)
            for r in remaining:
                current_mask[r] = True
            mask_t = torch.from_numpy(current_mask).bool().unsqueeze(0)

            probs, _ = policy(obs_t, mask_t)
            mask_probs = probs.squeeze(0).cpu().numpy()
            remaining_probs = mask_probs[remaining]
            psum = remaining_probs.sum()
            if psum > 0:
                remaining_probs = remaining_probs / psum
            else:
                remaining_probs = np.ones_like(remaining_probs) / len(remaining_probs)
            idx = remaining[np.random.choice(len(remaining), p=remaining_probs)]
            actions.append(idx)
            remaining.remove(idx)

    return actions
