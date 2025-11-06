import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path("/content/drive/MyDrive/Hedera Mood Oracle")
DATA_DIR = BASE_DIR / "data"

with open(DATA_DIR / "latest_prediction.json") as f:
    pred = json.load(f)

score = pred["predicted_score"]
mood = pred["predicted_class"]
confidence = pred.get("confidence", "HAUTE")

texts = {
    "Extreme Fear": "Les marchés tremblent. Les cœurs battent à se rompre. Une peur viscérale s'empare des âmes. Et pourtant... c'est peut-être le moment d'acheter.",
    "Fear": "L'incertitude plane. Les mains hésitent. Le doute s'installe. Mais la peur est souvent le dernier signal avant le retournement.",
    "Neutral": "Calme plat. Équilibre précaire. Ni euphorie, ni panique. Le marché respire, en attendant le prochain souffle.",
    "Greed": "L'appétit grandit. Les yeux brillent. Les mains s'ouvrent pour saisir. La cupidité murmure : 'encore un peu plus'.",
    "Extreme Greed": "Folie collective. Euphorie dévorante. Tout le monde achète. Les sommets semblent infinis. Attention : les chutes sont brutales."
}

nft = {
    "name": f"Hedera Mood Oracle #{pd.Timestamp.now().strftime('%Y-%m-%d')}",
    "description": texts[mood],
    "image": f"ipfs://TODO/{mood.lower().replace(' ', '_')}.png",
    "attributes": [
        {"trait_type": "Mood", "value": mood},
        {"trait_type": "FearGreed Index", "value": score},
        {"trait_type": "Confidence", "value": confidence},
        {"display_type": "date", "trait_type": "Updated", "value": int(pd.Timestamp.now().timestamp())}
    ]
}

with open(DATA_DIR / "nft_metadata.json", 'w') as f:
    json.dump(nft, f, indent=2)

print(f"MOOD : {mood} ({score}) [{confidence}]")
print("nft_metadata.json généré !")
