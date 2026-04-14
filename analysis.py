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


MODIFIED_PATHS = {
    "low": Path("results/gpt-5.4_low_eval.json"),
    "med": Path("results/gpt-5.4_med_eval.json"),
    "none": Path("results/gpt-5.4_none_eval.json"),
}


def _load_results(path):
    with open(path) as f:
        return json.load(f)["results"]


def text_modification(base_path=INPUT_PATH, modified_paths=MODIFIED_PATHS):
    """For each modified dataset, report how base-dataset U2 samples are
    re-classified after the text modification.
    """
    base_results = _load_results(base_path)
    u2_ids = [sid for sid, s in base_results.items() if classify(s) == "U2"]
    n_u2 = len(u2_ids)
    print(f"Base U2 samples: {n_u2}")

    summary = {}
    for name, path in modified_paths.items():
        mod_results = _load_results(path)
        transitions = {"R": 0, "U1": 0, "U2": 0, "S": 0, "error": 0, "missing": 0}
        for sid in u2_ids:
            if sid not in mod_results:
                transitions["missing"] += 1
                continue
            transitions[classify(mod_results[sid])] += 1

        print(f"\n[{name}] transitions from base U2 (n={n_u2}):")
        for k in ["R", "U1", "U2", "S", "error"]:
            pct = 100 * transitions[k] / n_u2 if n_u2 else 0
            print(f"  U2 -> {k}: {transitions[k]} ({pct:.2f}%)")
        if transitions["missing"]:
            print(f"  missing in modified set: {transitions['missing']}")

        # On the U2 subset, base error rate is 0 by construction.
        mod_error_rate = 100 * transitions["error"] / n_u2 if n_u2 else 0
        print(f"  change in error rate (on U2 subset): {mod_error_rate:+.2f} pp")
        summary[name] = transitions

    return summary


def s_created(base_path=INPUT_PATH, modified_paths=MODIFIED_PATHS):
    r"""Compute S_created = max(0, S_mod - S_base) / (1 - S_base) per modified set.

    S_dataset = |Synergistic samples| / |D|.
    """
    base_results = _load_results(base_path)
    base_total = len(base_results)
    base_syn = sum(1 for s in base_results.values() if classify(s) == "S") # count the number of synergistic samples in the unmod data.
    s_base = base_syn / base_total if base_total else 0.0 # express s_base as a fraction
    print(f"S_base = {s_base:.4f}  ({base_syn}/{base_total})")

    scores = {}
    for name, path in modified_paths.items():
        mod_results = _load_results(path)
        mod_total = len(mod_results)
        mod_syn = sum(1 for s in mod_results.values() if classify(s) == "S")
        s_mod = mod_syn / mod_total if mod_total else 0.0

        delta = s_mod - s_base
        denom = 1 - s_base
        score = max(0.0, delta) / denom if denom > 0 else 0.0
        print(f"[{name}] S_mod = {s_mod:.4f}  ({mod_syn}/{mod_total})  "
              f"ΔS = {delta:+.4f}  S_created = {score:.4f}")
        scores[name] = {"s_mod": s_mod, "delta": delta, "s_created": score}

    return {"s_base": s_base, "modified": scores}


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

    print("\n=== text_modification ===")
    text_modification()
    print("\n=== s_created ===")
    s_created()


if __name__ == "__main__":
    main()
