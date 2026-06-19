from kaggle_environments import make
from main import agent, DECK
import json

env = make("cabt", configuration={"decks": [DECK, DECK]})
env.run([agent, agent])

# Extract observation from first non-deck-select step
for step in env.steps:
    for p in step:
        obs = p.observation
        if obs and obs.get("select") is not None and obs.get("current") is not None:
            print("=== OBSERVATION STRUCTURE ===")
            print(json.dumps(obs, indent=2, default=str)[:3000])
            print()
            print("=== SELECT KEYS ===")
            print(list(obs["select"].keys()))
            print(f"  option count: {len(obs['select']['option'])}")
            print(f"  maxCount: {obs['select'].get('maxCount')}")
            print(f"  minCount: {obs['select'].get('minCount')}")
            print()
            print("=== CURRENT KEYS ===")
            print(list(obs["current"].keys()))
            print()
            print("=== PLAYER 0 KEYS ===")
            print(list(obs["current"]["players"][0].keys()))
            print()
            print("=== PLAYER 0 HAND ===")
            hand = obs["current"]["players"][0].get("hand", [])
            print(f"  hand count: {len(hand)}")
            if hand:
                print(f"  first card keys: {list(hand[0].keys()) if isinstance(hand[0], dict) else 'not dict'}")
            print()
            print("=== PLAYER 1 (opponent) KEYS ===")
            print(list(obs["current"]["players"][1].keys()))
            break
    else:
        continue
    break
