# Traffic Signs Recognition (GTSRB) – Keras/TensorFlow

A production‑ready, research‑grade implementation of **German Traffic Sign Recognition Benchmark (GTSRB)** classification using **Keras (TensorFlow 2.x)**. The project follows a clean, professional Python package structure (PEP 621 via `pyproject.toml`) with CI, linting, tests, and reproducible training.

> **Dataset**: Kaggle – *GTSRB – German Traffic Sign Recognition Benchmark*  
> https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

---

## 📦 Features
- End‑to‑end pipeline: data loading, augmentation, training, evaluation, and inference
- Config‑driven (single source of truth in `config.py`)
- Mixed precision, cosine LR schedules, early stopping, and model checkpointing
- Clear experiment artifacts structure under `runs/<exp_name>`
- Lightweight unit tests with `pytest`
- CI with Ruff + Black + Pytest (GitHub Actions)

---

## 🗂 Repository Structure
```
traffic-signs-cnn/
├─ src/
│  └─ traffic_signs/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ data.py
│     ├─ augmentations.py
│     ├─ model.py
│     ├─ callbacks.py
│     ├─ train.py
│     ├─ evaluate.py
│     └─ infer.py
├─ tests/
│  ├─ test_config.py
│  └─ test_model.py
├─ .github/workflows/ci.yml
├─ .editorconfig
├─ .gitignore
├─ LICENSE
├─ pyproject.toml
└─ README.md
```

---

## 🚀 Quickstart

### 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

### 2) Get the dataset
1. Download **GTSRB** from Kaggle (link above).  
2. Unzip to a local path and set an env var or pass a CLI argument:
   - Environment variable (recommended):
     ```bash
     export GTSRB_DIR=/path/to/GTSRB
     ```
   - or pass `--data_dir` to commands below.

**Expected layout** (typical for this Kaggle dataset):
```
GTSRB/
├─ Train/
│  ├─ 00000/  # class 0
│  ├─ 00001/
│  └─ .../    # class 42
└─ Test/
   ├─ 00000/
   ├─ 00001/
   └─ .../
```

> If your layout differs, you can still point to custom subfolders with `--train_dir` / `--val_dir` / `--test_dir`.

### 3) Train
```bash
python -m traffic_signs.train \
  --data_dir "$GTSRB_DIR" \
  --epochs 30 \
  --batch_size 64 \
  --img_size 48 \
  --exp_name baseline
```

### 4) Evaluate
```bash
auth_token=$(python -c "import secrets;print(secrets.token_hex(4))")
python -m traffic_signs.evaluate \
  --data_dir "$GTSRB_DIR" \
  --checkpoint runs/baseline/checkpoints/best.h5 \
  --batch_size 128
```

### 5) Inference on an image or folder
```bash
python -m traffic_signs.infer \
  --checkpoint runs/baseline/checkpoints/best.h5 \
  --input /path/to/image_or_dir \
  --class_map_json runs/baseline/class_map.json
```

---

## ⚙️ Configuration
All defaults live in `src/traffic_signs/config.py`. You can override via CLI.

Key options:
- `img_size`: input resolution (default 48×48)
- `augment`: enable/disable albumentations‑like Keras preprocessing
- `mixed_precision`: set `--mixed_precision` for faster training on modern GPUs
- `optimizer`: AdamW with cosine decay by default

---

## 🧠 Model
Default is a compact **CNN** (Conv‑BN‑ReLU blocks + Squeeze‑and‑Excitation + Dropout). It reaches strong baseline accuracy on GTSRB with <2M params and fast inference. You can swap architecture via `build_cnn(...)` args.

---

## 🧪 Tests
```bash
pytest -q
```

---

## 🧰 Developer Tooling
- **Lint**: `ruff check .`  
- **Format**: `ruff format .`  
- **Type hints**: `mypy` optional (add if desired)

---

## 📈 Expected Results (baseline)
- Top‑1 acc: **>97%** on the validation split with 48×48 images and moderate augments (varies by split & seed)

> Reproducibility is subject to nondeterminism of GPU ops and data order. Use `--seed` to improve determinism.

---

## 📜 License
This project is released under the **MIT License**. See `LICENSE` for details.

---

## 🙌 Credits
- GTSRB authors & dataset contributors
- Keras / TensorFlow team

---

## 🔗 Links
- Dataset: Kaggle – GTSRB (German Traffic Sign)  
- Paper: *The German Traffic Sign Recognition Benchmark: A multi-class classification competition* (Stallkamp et al.)

