import json
from pathlib import Path

# TODO: change these paths for each (modified) dataset!
INPUT_PATH = Path("results/base_dataset.json")
ERROR_PATH = Path("results/erroneous.json")


def classify(sample):
    y = sample["ground_truth"]
    x1 = sample["image_only"]
    x2 = sample["text_only"]
    x12 = sample["multimodal"]

    x1_c = x1 == y
    x2_c = x2 == y
    x12_c = x12 == y

    if x1_c and x2_c and x12_c:
        return "R"
    if x1_c and not x2_c and x12_c:
        return "U1"
    if x2_c and not x1_c and x12_c:
        return "U2"
    if not x1_c and not x2_c and x12_c:
        return "S"
    return "error"


def main():
    with open(INPUT_PATH) as f:
        data = json.load(f)

    results = data["results"]
    counts = {"R": 0, "U1": 0, "U2": 0, "S": 0, "error": 0}
    errors = {}

    for sid, sample in results.items():
        label = classify(sample)
        counts[label] += 1
        if label == "error":
            errors[sid] = sample
        else:
            sample["interaction"] = label

    total = len(results)
    print(f"Total samples: {total}")
    for k in ["R", "U1", "U2", "S", "error"]:
        pct = 100 * counts[k] / total if total else 0
        print(f"{k}: {counts[k]} ({pct:.2f}%)")

    with open(ERROR_PATH, "w") as f:
        json.dump(errors, f, indent=2)
    print(f"\nSaved {len(errors)} erroneous samples to {ERROR_PATH}")

    with open(INPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Updated {INPUT_PATH} with interaction labels")


if __name__ == "__main__":
    main()
