import re

with open(".venv/Lib/site-packages/kaggle_environments/envs/cabt/cabt.py", "r") as f:
    content = f.read()

# Find deck = [...] 
match = re.search(r"deck = (\[[\d,\s\n]+\])", content)
if match:
    deck = eval(match.group(1))
    print(f"Deck length: {len(deck)}")
    with open("deck.csv", "w") as f:
        for cid in deck:
            f.write(f"{cid}\n")
    print("deck.csv written successfully")
else:
    print("Could not find deck in cabt.py")
