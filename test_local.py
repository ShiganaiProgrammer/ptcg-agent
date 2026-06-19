from kaggle_environments import make
from main import agent, _load_deck

DECK = _load_deck()
env = make("cabt", configuration={"decks": [DECK, DECK]})
env.run([agent, agent])

print(f"Total steps: {len(env.steps)}")
print(f"Done: {env.done}")
print()

for i, step in enumerate(env.steps[:5]):
    print(f"Step {i}:")
    for j, p in enumerate(step):
        status = p.status
        reward = p.reward
        action = p.action
        has_select = p.observation.get("select") is not None if p.observation else None
        print(f"  Player {j}: status={status}, reward={reward}, action_len={len(action) if action else 0}, has_select={has_select}")

with open("result.html", "w", encoding="utf-8") as f:
    f.write(env.render(mode="html"))
print("\nResult HTML saved.")
