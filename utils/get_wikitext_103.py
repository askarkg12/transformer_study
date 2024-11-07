from pathlib import Path
from datasets import load_dataset


TEXT_FILE = Path("big_data/all_text.txt")
TEXT_FILE.parent.mkdir(parents=True, exist_ok=True)

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

splits = ds.keys()

split_texts = []
for split in splits:
    data = ds[split]
    split_texts.append("".join(data["text"]))
text = "".join(split_texts)

with open(TEXT_FILE, "w", encoding="utf-8") as file:
    file.write(text)
pass
