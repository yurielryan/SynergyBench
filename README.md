# SynergyBench

SynergyBench is a lightweight evaluation pipeline for multimodal sarcasm classification.

Given text and image pairs, it runs three inference modes:
- text-only
- image-only
- multimodal (text + image)

The default backend is Qwen3-VL via Hugging Face Transformers.

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
- This repo provides the core inference pipeline to obtain text-only, image-only,
  and multimodal predictions needed for synergy computation.
- You can then compute $S_{base}$ and $S_{mod}$ downstream from the saved outputs.

## Dataset Source

This project uses data from **DocMSU**:
- https://github.com/fesvhtr/DocMSU

Please follow DocMSU's license and usage terms when downloading and redistributing data.

## Repository Structure

- `evaluate.py`: main evaluation entrypoint.
- `curate_dataset.py`: deterministic split curation script.
- `models/qwen.py`: Qwen3-VL local inference wrapper.
- `models/openai.py`: OpenAI API wrapper.
- `models/gemini.py`: Gemini API wrapper.
- `models/llama.py`: Llama text-model wrapper.
- `models/llava.py`: LLaVA multimodal wrapper.
- `config.yaml`: evaluation configuration.
- `docmsu_all.json`: full JSON dataset mapping (`sample_id -> sample`).
- `docmsu_2500_split.json`: curated balanced split file.
- `img/`: image files referenced by `img_name`.

## Requirements

- Linux/macOS (tested on Linux)
- Python 3.10+
- CUDA GPU recommended for Qwen3-VL inference

Python dependencies:
- `torch`
- `transformers`
- `Pillow`
- `PyYAML`
- `accelerate` (recommended when using `device_map="auto"`)
- `openai` (for OpenAI API backend)
- `google-generativeai` (for Gemini API backend)

## 1) Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers pillow pyyaml accelerate openai google-generativeai
```

If you are using gated/private models on Hugging Face, authenticate first:

```bash
huggingface-cli login
```

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

## 3) Recreate the Curated 2500-Sample Split (Optional)

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

Edit `config.yaml` as needed.

Default configuration:
- model provider: `qwen`
- model size: `8b` (resolves to `Qwen/Qwen3-VL-8B-Instruct`)
- dataset: `docmsu_2500_split.json`
- split: `all`
- image directory: `img`
- output path: `results/inference.json`

### Useful Config Options

- `model.model_id`: set exact Hugging Face model ID (overrides `size`)
- `model.max_new_tokens`: generation length (default is small for yes/no output)
- `model.system_prompt`: classification instruction prompt
- `data.split`: one of `all`, `train`, `validation`, `test`

## Model Backends

`evaluate.py` now supports these `model.provider` values:
- `qwen` (local)
- `openai` (API)
- `gemini` (API)
- `llama` (local text model)
- `llava` (local multimodal model)

### Example: OpenAI

```yaml
model:
  provider: openai
  model_id: gpt-4.1-mini
  max_new_tokens: 8
  system_prompt: "You are a sarcasm classifier. Answer with exactly one token: yes or no."
  # optional if OPENAI_API_KEY is already exported
  # api_key: "..."
```

### Example: Gemini

```yaml
model:
  provider: gemini
  model_id: gemini-2.0-flash
  max_new_tokens: 8
  temperature: 0.0
  system_prompt: "You are a sarcasm classifier. Answer with exactly one token: yes or no."
  # optional if GEMINI_API_KEY is already exported
  # api_key: "..."
```

Set API credentials as environment variables when using API-based wrappers:

```bash
export OPENAI_API_KEY="your_openai_key"
export GEMINI_API_KEY="your_gemini_key"
```

## 5) Run Evaluation

```bash
python evaluate.py --config config.yaml
```

On success, the script writes:

- `results/inference.json`

## 6) Output Format

The output JSON contains:
- a copy of the config used
- per-sample predictions:
  - `ground_truth`
  - `text_only`
  - `image_only`
  - `multimodal`

Example shape:

```json
{
  "config": {"...": "..."},
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

## Reproducibility Notes

- Split generation is deterministic with the fixed seed (`20260311`).
- The curation script sorts IDs before shuffling to stabilize behavior across JSON key orders.
- For strict reproducibility across machines, pin package versions and record the exact model revision from Hugging Face.

## Troubleshooting

- Out-of-memory errors:
  - Use a smaller model or quantized variant
  - Reduce concurrent GPU load
- Missing image predictions (`unknown` in image mode):
  - Verify `data.image_dir` and `img_name` path alignment
- Slow startup:
  - First run downloads model weights from Hugging Face, which can be large

## Citation and Attribution

If you use this pipeline, please cite/credit:
- DocMSU dataset: https://github.com/fesvhtr/DocMSU
- Qwen model family (when applicable)
