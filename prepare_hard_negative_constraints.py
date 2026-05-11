import json
from pathlib import Path
from urllib.request import urlopen, Request
import re
from datasets import load_dataset
import numpy as np
import os


def get_bad_words_constraints():
    request = Request(
        "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en",
        headers={"User-Agent": "python-urlopen"},
    )
    with urlopen(request, timeout=30) as response:
        text = response.read().decode("utf-8")
    words = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    
    output_path = Path("./predefined_refusal_keywords/refusal_toxic_words_en.json")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(words)} words to {output_path}")


def get_enron_constraints():
    # Loading dataset
    dataset = load_dataset("LLM-PBE/enron-email", split="train")
    regex_patterns = [
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Email
        r"[0-9]{3}-[0-9]{2}-[0-9]{4}",                      # SSN-like
        r"[0-9]{3}-[0-9]{3}-[0-9]{4}",                      # Phone-like
        r"[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{4}"              # Credit Card-like
    ]
    combined_regex = re.compile("|".join(regex_patterns))
    
    def contains_pii(data):
        text = data.get("text", "")
        if text:
            match = re.search(combined_regex, text)
            if match:
                start_idx = match.start()
                prompt = text[:start_idx]
                if len(prompt) >= 20 and len(prompt)<=2000:
                    return True
        return False

    dataset = dataset.filter(contains_pii, num_proc=4)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(dataset), size=min(500, len(dataset)), replace=False)
    dataset = dataset.select(idx.tolist())

    # Save matched spans from the final sampled examples.
    matched_spans = set()
    for data in dataset:
        text = data.get("text", "")
        for match in combined_regex.finditer(text):
            matched_spans.add(match.group(0))

    output_path = "./predefined_refusal_keywords/refusal_enron.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted(matched_spans), f, ensure_ascii=False, indent=2)

    print(f"Saved {len(matched_spans)} Enron matched spans to {output_path}")


def main():
    os.makedirs("./predefined_refusal_keywords/", exist_ok=True)
    get_bad_words_constraints()
    get_enron_constraints()


if __name__ == "__main__":
    main()