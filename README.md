# SynergyBench

SynergyBench is a lightweight evaluation pipeline for multimodal sarcasm classification.

Given text and image pairs, it runs three inference modes:
- text-only
- image-only
- multimodal (text + image)

The default evaluator backend is OpenAI-compatible via OpenRouter.

## Benchmark Goal

The goal of SynergyBench is to measure whether a model can create **synergistic multimodal information**.

Given two modalities $X_1$ (text) and $X_2$ (image), we compare unimodal and
multimodal predictions to classify interaction type:

| Multimodal Interaction | $X_1$ | $X_2$ | $X_1 \cup X_2$ |
|---|---|---|---|
| Redundant | $y$ | $y$ | $y$ |
| $X_1$ Unique | $y$ | $\neg y$ | $y$ |
| $X_2$ Unique | $\neg y$ | $y$ | $y$ |
| Synergistic | $\neg y$ | $\neg y$ | $y$ |

Interpretation:
- **Redundant**: either modality alone is sufficient.
- **Unique**: only one modality carries task-relevant information.
- **Synergistic**: neither modality alone is enough, but their combination is.

## Proposed Task: Measuring Synergy Creation

We measure how much synergy is created when transforming a baseline dataset
$\mathcal{D}$ into a modified dataset $\mathcal{D'}$.

- **Input**: baseline dataset $\mathcal{D}$ and modified dataset $\mathcal{D'}$.
- **Output**: synergy creation score $S_{created}$.

Scoring:

$$
\Delta S = S_{mod} - S_{base}
$$

$$
S_{created} = \frac{\Delta S}{1 - S_{base}}
$$

where:
- $S_{base}$ is the synergy score computed on $\mathcal{D}$
- $S_{mod}$ is the synergy score computed on $\mathcal{D'}$

Current repository status:
- `evaluate.py` provides the core inference pipeline to obtain text-only,
  image-only, and multimodal predictions needed for synergy computation.
- `analysis.py` classifies each sample into interaction categories
  (`R`, `U1`, `U2`, `S`, `error`) and computes $S_{base}$, $S_{mod}$,
  $\Delta S$, and $S_{created}$ directly from saved result JSONs.

## Dataset Source

This project uses data from **DocMSU**:
- https://github.com/fesvhtr/DocMSU

Please follow DocMSU's license and usage terms when downloading and redistributing data.

## Repository Structure

```text
SynergyBench/
├── evaluate.py                         # Main evaluation script
├── analysis.py                         # Interaction classification + synergy metrics
├── curate_dataset.py                   # Split curation - train/val/test
├── docmsu_all.json                     # Base dataset
├── docmsu_2500_split.json              # Curated balanced split file
├── img/
├── results/                            # Evaluation outputs + checkpoints
├── evaluator_models/
│   ├── base_model_evaluation.py        # Abstract evaluator interface
│   ├── openai.py                       # Example implementation
│   └── utils.py                        # Shared helpers
└── generator_models/                   # Generation model wrappers (separate from evaluator pipeline)
```

## Requirements

- Linux/macOS (tested on Linux)
- Python 3.10+
- OpenRouter or OpenAI API key

Python dependencies:
- `openai` (for OpenAI API backend)

## 1) Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install openai
```

Set an API key before running evaluation:

```bash
export OPENROUTER_API_KEY="your_key_here"
# or:
export OPENAI_API_KEY="your_key_here"
```

Alternatively, set your api key in a .env file with the same names variable names above.

## 2) Prepare Data (DocMSU)

1. Download DocMSU assets from:
   - https://github.com/fesvhtr/DocMSU
2. Place the JSON and images so this repo has:
   - `docmsu_all.json`
   - `img/` containing image files like `technology_00727.jpg`

Expected sample schema in `docmsu_all.json`:

```json
{
  "sample_id": {
    "is_sar": 0,
    "text": "...",
    "img_name": "example.jpg",
    "type": "technology"
  }
}
```

## 3) Recreate the Curated 2500-Sample Split

To regenerate the balanced deterministic split used in this repo:

```bash
python curate_dataset.py \
  --input docmsu_all.json \
  --output docmsu_2500_split.json \
  --seed 20260311
```

What this does:
- samples exactly 2500 items
- enforces 1250 non-sarcastic (`is_sar=0`) and 1250 sarcastic (`is_sar=1`)
- creates `train/validation/test` splits in 80/10/10 ratio
- writes metadata and split counts under `meta`

## 4) Configure Evaluation

Evaluation is configured through CLI flags in `evaluate.py`.

Default configuration:
- model provider: `openai`
- dataset: `docmsu_2500_split.json`
- split: `all`
- image directory: `img`
- output path: `results/evaluator_inference.json`

### Useful CLI Options

- `--dataset`: path to dataset JSON (raw mapping or curated split JSON)
- `--split`: one of `all`, `train`, `validation`, `test`
- `--image-dir`: directory containing image files referenced by `img_name`
- `--output`: output JSON path
- `--provider`: evaluator provider (currently `openai`)
- `--limit`: optional sample limit for quick smoke tests
- `--modes`: comma-separated subset of `text_only,image_only,multimodal`
  to run (default: all three)
- `--checkpoint-every`: save partial results every N evaluations
  (default `250`; `0` disables checkpointing)
- `--resume`: resume from an existing checkpoint at `--output`
- `--text-override`: path to a generated-text JSON
  (`{"response": {sample_id: {"context": "..."}}}`) whose `context`
  replaces each sample's `text` before evaluation — used to evaluate
  modified-text datasets $\mathcal{D'}$
- `--image-results-from`: reuse `image_only` predictions from a prior
  results JSON (skips re-running image-only on unchanged images)
- `--text-results-from`: mirror of the above for `text_only`
- `--filter-interactions`: comma-separated labels (e.g. `U2` or `U1,S`)
  to keep; requires `--interaction-source` (or `--text-results-from`)
  to point at a results JSON with per-sample `interaction` fields
  (populated by `analysis.py`)
- `--interaction-source`: explicit path to the results JSON used for
  `--filter-interactions` (defaults to `--text-results-from`)

## Model Backends

`evaluate.py` currently supports:
- `openai` (API via OpenRouter-compatible OpenAI SDK)

Additional evaluator providers can be added in `evaluator_models/` and wired in `init_evaluator_model()`.



## 5) Run Evaluation

```bash
python evaluate.py
```

Example quick smoke test:

```bash
python evaluate.py --split validation --limit 5 --output results/smoke_eval.json
```

On success, the script writes:

- `results/evaluator_inference.json` (or your custom `--output` path)

## 6) Output Format

The output JSON contains:
- a copy of run arguments under `run_config`
- per-sample predictions:
  - `ground_truth`
  - `text_only`
  - `image_only`
  - `multimodal`

Example shape:

```json
{
  "run_config": {"...": "..."},
  "results": {
    "technology_00727": {
      "sample_id": "technology_00727",
      "ground_truth": "no",
      "text_only": "no",
      "image_only": "unknown",
      "multimodal": "no"
    }
  }
}
```

## 7) Analyze Interactions and Synergy Creation

Once you have at least a base results JSON (and optionally one or more
modified-dataset results JSONs), run `analysis.py` to classify each
sample and compute synergy creation scores.

Configure the paths at the top of `analysis.py`:

```python
INPUT_PATH = Path("results/base_dataset.json")
ERROR_PATH = Path("results/erroneous.json")

MODIFIED_PATHS = {
    "low":  Path("results/gpt-5.4_low_eval.json"),
    "med":  Path("results/gpt-5.4_med_eval.json"),
    "none": Path("results/gpt-5.4_none_eval.json"),
}
```

Then run:

```bash
python analysis.py
```

What `analysis.py` does:

- **`classify(sample)`**: maps each `{ground_truth, text_only,
  image_only, multimodal}` tuple to one of:
  - `R`  — redundant (both unimodal correct, multimodal correct)
  - `U1` — image-only unique (image correct, text wrong, multimodal correct)
  - `U2` — text-only unique (text correct, image wrong, multimodal correct)
  - `S`  — synergistic (both unimodal wrong, multimodal correct)
  - `error` — anything else (e.g. multimodal wrong)
- Writes `interaction` labels back into `INPUT_PATH` for every
  non-`error` sample (enabling `--filter-interactions` in `evaluate.py`).
- Dumps all `error` samples to `ERROR_PATH`.
- **`text_modification(...)`**: for each modified results JSON, reports
  how base-dataset `U2` samples are re-classified after text
  modification (useful for measuring synergy created by text edits).
- **`s_created(...)`**: computes
  $S_{base}$, $S_{mod}$, $\Delta S$, and
  $S_{created} = \max(0, \Delta S) / (1 - S_{base})$ for each
  modified dataset.

Typical workflow:

1. Run `evaluate.py` on the base dataset → `results/base_dataset.json`.
2. Run `evaluate.py` with `--text-override` on each modified-text JSON
   → `results/<variant>_eval.json`.
3. Update paths in `analysis.py` and run it to label interactions and
   print synergy creation scores.

## Reproducibility Notes

- Split generation is deterministic with the fixed seed (`20260311`).
- The curation script sorts IDs before shuffling to stabilize behavior across JSON key orders.
- For strict reproducibility across machines, pin package versions and record the exact model revision from Hugging Face.

## Troubleshooting

- Missing API key errors:
  - Ensure `OPENROUTER_API_KEY` (preferred) or `OPENAI_API_KEY` is set
- Missing image predictions (`unknown` in image mode):
  - Verify `--image-dir` and `img_name` path alignment
  - Check that referenced image files exist under `img/`

## Citation and Attribution

If you use this pipeline, please cite/credit:
- DocMSU dataset: https://github.com/fesvhtr/DocMSU
- Qwen model family (when applicable)
