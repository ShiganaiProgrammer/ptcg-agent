from kaggle_environments import make
from main import agent, DECK
import json

env = make("cabt", configuration={"decks": [DECK, DECK]})
env.run([agent, agent])

# Find a mid-game observation with cards
for step in env.steps:
    for p in step:
        obs = p.observation
        if obs and obs.get("select") is not None and obs.get("current"):
            players = obs["current"]["players"]
            p0 = players[0]
            if p0.get("active") and len(p0["active"]) > 0 and p0["active"][0] is not None:
                print("=== MID-GAME OBSERVATION (sample) ===")
                # Print limited view
                current = obs["current"]
                print(f"turn: {current['turn']}, result: {current['result']}")
                print(f"supporterPlayed: {current['supporterPlayed']}, energyAttached: {current['energyAttached']}, retreated: {current['retreated']}")
                
                print("\n--- Player 0 (self) ---")
                print(f"handCount: {p0['handCount']}, deckCount: {p0['deckCount']}")
                print(f"active: {json.dumps(p0['active'], default=str)[:500]}")
                print(f"bench count: {len(p0['bench'])}")
                for i, b in enumerate(p0['bench']):
                    if b:
                        print(f"  bench[{i}]: {json.dumps(b, default=str)[:200]}")
                print(f"prize count: {len(p0['prize'])}")
                print(f"poisoned: {p0['poisoned']}, burned: {p0['burned']}, asleep: {p0['asleep']}")
                
                print("\n--- Player 1 (opponent) ---")
                p1 = players[1]
                print(f"handCount: {p1['handCount']}, deckCount: {p1['deckCount']}")
                print(f"active: {json.dumps(p1['active'], default=str)[:500]}")
                print(f"bench count: {len(p1['bench'])}")
                for i, b in enumerate(p1['bench']):
                    if b:
                        print(f"  bench[{i}]: {json.dumps(b, default=str)[:200]}")
                
                print("\n--- Select ---")
                sel = obs["select"]
                print(f"type: {sel['type']}, context: {sel['context']}")
                print(f"option count: {len(sel['option'])}")
                for i, opt in enumerate(sel['option'][:5]):
                    print(f"  option[{i}]: {json.dumps(opt, default=str)[:200]}")
                if len(sel['option']) > 5:
                    print(f"  ... and {len(sel['option']) - 5} more")
                break
    else:
        continue
    break
