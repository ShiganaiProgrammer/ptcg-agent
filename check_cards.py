from kaggle_environments import make

env = make("cabt")
cards = env.spec.configuration.all_card_data()
print(f"Total cards available: {len(cards)}")
print()

for i, card in enumerate(cards[:5]):
    cid = card.get("id")
    name = card.get("name")
    hp = card.get("hp")
    ctype = card.get("type")
    print(f"Card {i}: id={cid}, name={name}, hp={hp}, type={ctype}")

print()
card = cards[0]
for k, v in card.items():
    print(f"  {k}: {v}")
