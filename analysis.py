import json
from pathlib import Path

# NOTE: change these paths for each (modified) dataset!
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


def text_modification():
    # TODO: this function modifies a sample that is classified as U2 in the base dataset.
    # TODO 1: identify all samples that are U1.
    # TODO 2: for these identified samples, classify the interactions from these results - json files: generated_text/gpt-5.4_low.json, generated_text/gpt-5.4_med.json, generated_text/gpt-5.4_none.json
    # TODO 3: compare the change in classification from U1 to {R, U2, S} --> print out the percentage changed to redundancy, changed to U2, and changed to S. also print out the change in error.
    # TODO 4: print out the change in classifications for these samples.
    pass


def s_created():
    """We compute the overall synergy in a dataset using the taxonomy (Table~\ref{tab:taxonomy}) and definition in Section~\ref{sec:task_def}. For simplicity, we compute this term with the fraction of synergistic samples over the size (number of samples) of the entire dataset:
    
    $$\text{Overall Synergy} = \frac{|\text{Synergistic Samples}|}{|\mathcal{D}|} \in [0,1]$$
    With this expression, we can compute $S_{base}$ and $S_{mod}$: the overall synergy in the base and modified dataset (from our experiments) respectively. Subsequently, we assign a score, $S_{created}\in [0,1]$:
    
    $$\Delta S = S_{mod} - S_{base}\quad s.t.\ S_{mod}\ge S_{base}$$
    $S_{created} = \frac{\Delta S}{1 - S_{base}}$$"""
    
    # TODO: this function computes the amount of synergy created due to a modification.
    pass


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
