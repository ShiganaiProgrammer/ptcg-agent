"""Gymnasium wrapper for the cabt (Pokemon TCG) environment."""

from kaggle_environments.envs.cabt.cg.game import battle_start, battle_select, battle_finish
from kaggle_environments.envs.cabt.cg.sim import Battle
import numpy as np
import traceback

MAX_OPTIONS = 64
MAX_BENCH = 5
MAX_HAND = 10
MAX_PRIZE = 6
MAX_DISCARD = 30

CARD_TYPE_POKEMON = 1
CARD_TYPE_TRAINER = 2
CARD_TYPE_ENERGY = 3


class CabtEnv:
    """Wraps cabt engine into a Gymnasium-like interface for RL.
    
    Observations: dict with keys:
        - 'obs': flat numpy array encoding the board state
        - 'action_mask': boolean array of valid actions (size MAX_OPTIONS)
        - 'select_type': int, type of the current selection
        - 'max_count': int, max number of options to select
    
    Actions: numpy array of int indices into the select option list.
    """
    
    def __init__(self, deck_self, deck_opponent=None, max_episode_steps=1000):
        self.deck_self = deck_self
        self.deck_opponent = deck_opponent if deck_opponent is not None else deck_self.copy()
        self.max_episode_steps = max_episode_steps
        self._step_count = 0
        self._done = False
    
    def reset(self):
        """Start a new battle. Returns initial observation."""
        battle_finish()
        obs, start_data = battle_start(self.deck_self, self.deck_opponent)
        if start_data.errorPlayer >= 0:
            raise RuntimeError(f"Deck error for player {start_data.errorPlayer}: type={start_data.errorType}")
        self._step_count = 0
        self._done = False
        return self._process_obs(Battle.obs)
    
    def step(self, action):
        """Apply one action. Returns (obs, reward, done, info)."""
        if self._done:
            raise RuntimeError("Episode already done. Call reset().")
        
        if isinstance(action, np.ndarray):
            action = action.tolist()
        elif isinstance(action, int):
            action = [action]
        
        try:
            battle_select(action)
        except IndexError as e:
            print(f"DEBUG: battle_select failed. select_type={Battle.obs.get('select', {}).get('type')}, "
                  f"option_count={len(Battle.obs.get('select', {}).get('option', []))}, "
                  f"maxCount={Battle.obs.get('select', {}).get('maxCount')}, "
                  f"minCount={Battle.obs.get('select', {}).get('minCount')}, "
                  f"action={action}")
            raise
        
        self._step_count += 1
        
        obs = Battle.obs
        current = obs.get("current")
        reward = 0.0
        done = False
        
        if current is not None and current.get("result", -1) >= 0:
            done = True
            result = current["result"]
            your_index = current.get("yourIndex", 0)
            if result == your_index:
                reward = 1.0
            elif result == 1 - your_index:
                reward = -1.0
            else:
                reward = 0.0
        
        if self._step_count >= self.max_episode_steps:
            done = True
        
        self._done = done
        return self._process_obs(obs), reward, done, {}
    
    def _process_obs(self, obs):
        """Convert raw cabt observation to a flat feature vector."""
        current = obs.get("current")
        select = obs.get("select")
        
        features = []
        your_idx = current.get("yourIndex", 0) if current else 0
        opponent_idx = 1 - your_idx
        players = current.get("players", [None, None]) if current else [None, None]
        me = players[your_idx] if players[your_idx] is not None else {}
        opp = players[opponent_idx] if players[opponent_idx] is not None else {}
        
        # === Game state features ===
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
        
        # === Self features ===
        # Active Pokemon
        active_list = me.get("active", [])
        if active_list and active_list[0] is not None:
            poke = active_list[0]
            features.append(poke.get("hp", 0) / 200.0)
            features.append(poke.get("maxHp", 0) / 200.0)
            features.append(len(poke.get("energies", [])) / 10.0)
            features.append(float(poke.get("appearThisTurn", False)))
            features.append(1.0 if poke.get("preEvolution") else 0.0)
        else:
            features.extend([0.0] * 5)
        
        # Bench Pokemon (up to MAX_BENCH)
        bench = me.get("bench", [])
        for i in range(MAX_BENCH):
            if i < len(bench) and bench[i] is not None:
                poke = bench[i]
                features.append(poke.get("hp", 0) / 200.0)
                features.append(len(poke.get("energies", [])) / 10.0)
                features.append(1.0 if poke.get("preEvolution") else 0.0)
            else:
                features.extend([0.0] * 3)
        features.append(len(bench) / 5.0)
        
        # Hand summary
        hand = me.get("hand", [])
        features.append(min(len(hand), MAX_HAND) / 10.0)
        pokemon_in_hand = sum(1 for c in hand if isinstance(c, dict) and c.get("cardType") == CARD_TYPE_POKEMON) / 10.0
        trainer_in_hand = sum(1 for c in hand if isinstance(c, dict) and c.get("cardType") == CARD_TYPE_TRAINER) / 10.0
        energy_in_hand = sum(1 for c in hand if isinstance(c, dict) and c.get("cardType") == CARD_TYPE_ENERGY) / 10.0
        features.extend([pokemon_in_hand, trainer_in_hand, energy_in_hand])
        
        # Deck count
        features.append(me.get("deckCount", 0) / 60.0)
        
        # Prize cards
        prize = me.get("prize", [])
        prize_remaining = sum(1 for p in prize if p is not None)
        features.append(prize_remaining / 6.0)
        
        # Status conditions
        features.extend([
            float(me.get("poisoned", False)),
            float(me.get("burned", False)),
            float(me.get("asleep", False)),
            float(me.get("paralyzed", False)),
            float(me.get("confused", False)),
        ])
        
        # === Opponent features ===
        # Active Pokemon
        opp_active_list = opp.get("active", [])
        if opp_active_list and opp_active_list[0] is not None:
            poke = opp_active_list[0]
            features.append(poke.get("hp", 0) / 200.0)
            features.append(len(poke.get("energies", [])) / 10.0)
            features.append(1.0 if poke.get("preEvolution") else 0.0)
        else:
            features.extend([0.0] * 3)
        
        # Opponent bench
        opp_bench = opp.get("bench", [])
        for i in range(MAX_BENCH):
            if i < len(opp_bench) and opp_bench[i] is not None:
                poke = opp_bench[i]
                features.append(poke.get("hp", 0) / 200.0)
                features.append(len(poke.get("energies", [])) / 10.0)
            else:
                features.extend([0.0] * 2)
        features.append(len(opp_bench) / 5.0)
        
        # Opponent info (visible)
        features.append(opp.get("handCount", 0) / 10.0)
        features.append(opp.get("deckCount", 0) / 60.0)
        opp_prize = opp.get("prize", [])
        opp_prize_remaining = sum(1 for p in opp_prize if p is not None)
        features.append(opp_prize_remaining / 6.0)
        
        # Opponent status
        features.extend([
            float(opp.get("poisoned", False)),
            float(opp.get("burned", False)),
            float(opp.get("asleep", False)),
            float(opp.get("paralyzed", False)),
            float(opp.get("confused", False)),
        ])
        
        obs_array = np.array(features, dtype=np.float32)
        
        # === Action mask ===
        action_mask = np.zeros(MAX_OPTIONS, dtype=bool)
        if select and select.get("option"):
            n_options = min(len(select["option"]), MAX_OPTIONS)
            action_mask[:n_options] = True
            select_type = select.get("type", -1)
            max_count = select.get("maxCount", 1)
        else:
            select_type = -1
            max_count = 0
        
        return {
            "obs": obs_array,
            "action_mask": action_mask,
            "select_type": select_type,
            "max_count": max_count,
        }
    
    def close(self):
        battle_finish()
