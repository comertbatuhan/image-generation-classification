# Reproducing Results

This repository reproduces the dataset generation and model training pipeline for **synthetic image classification** using Stable Diffusion v1.5 and logistic regression classifiers.

---
You must have:
- metadata.jsonl (provided)
- helper scripts (utils.py, preprocess_train_helpers.py, feature_extraction.py)
- access to GPU

## 1. Setup

### Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## 2. Regenerate Images from Metadata
```bash
from diffusers import StableDiffusionPipeline
import torch, json, os
from tqdm import tqdm

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

pipe.enable_attention_slicing()
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception as e:
    print("xFormers not enabled:", e)

os.makedirs("images_repro", exist_ok=True)

with open("metadata.jsonl", "r", encoding="utf-8") as f:
    for line in tqdm(f):
        rec = json.loads(line)
        seed = rec["seed"]
        prompt = rec["prompt"]
        out_path = f"images_repro/{os.path.basename(rec['image_path'])}"
        if os.path.exists(out_path):
            continue
        g = torch.Generator(device="cuda").manual_seed(seed)
        img = pipe(
            prompt,
            generator=g,
            height=rec["height"],
            width=rec["width"],
            num_inference_steps=rec["steps"],
            guidance_scale=rec["guidance"]
        ).images[0]
        img.save(out_path)

```

## 3. Feature Extraction
Extract HOG + LBP features from images using custom functions.
```bash
from utils import build_feature_matrix
import numpy as np

zip_paths = ["images.zip", "images2.zip", "images3.zip"]
X_all, paths_all = [], []

for zp in zip_paths:
    X, p = build_feature_matrix(zp)
    X_all.append(X)
    paths_all.append(p)

X = np.concatenate(X_all, axis=0)
paths = np.concatenate(paths_all, axis=0)
```

## 4. Merge Metadata and Features
```bash
import pandas as pd
import numpy as np

metadata = pd.read_json("metadata.jsonl", lines=True)
df_feats = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df_feats["path"] = paths
df = df_feats.merge(metadata, left_on="path", right_on="image_path", how="inner")
df = df[df["is_outlier"] == False]
```

## 5. Train Models
```bash
from preprocess_train_helpers import train_eval_logreg, build_X_num_only, build_X_cat_num
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
```
#### (a) Extracted Features Only
```bash
X_ext, y_ext = build_X_num_only(df, target_col='class')
y_ext = LabelEncoder().fit_transform(y_ext)
X_train, X_test, y_train, y_test = train_test_split(X_ext, y_ext, stratify=y_ext, random_state=42)
clf_ext, scaler_ext = train_eval_logreg(X_train, X_test, y_train, y_test)
```

#### (b) Categorical + Numerical Only
```bash
categorical_cols = ['color', 'season', 'origin', 'ripeness']
numerical_cols = ['weight_g', 'height', 'width']
X_catnum, y = build_X_cat_num(df, categorical_cols, numerical_cols, target_col='class')
y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_catnum, y, stratify=y, random_state=42)
clf_catnum, scaler_catnum = train_eval_logreg(X_train, X_test, y_train, y_test)
```

#### (c) Fused Dataset (All Features)
```bash
categorical_cols = ['color', 'season', 'origin', 'ripeness']
numerical_cols = [c for c in df.columns if c not in categorical_cols + ['class']]
X_fused, y = build_X_cat_num(df, categorical_cols, numerical_cols, target_col='class')
y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_fused, y, stratify=y, random_state=42)
clf_fused, scaler_fused = train_eval_logreg(X_train, X_test, y_train, y_test)
```

## 6. Expected Results
| Dataset                 | Accuracy  | Macro F1  | ROC AUC   |
| ----------------------- | --------- | --------- | --------- |
| Extracted Features      | 0.456     | 0.451     | 0.763     |
| Numerical + Categorical | 0.747     | 0.713     | 0.940     |
| Fused Dataset           | **0.805** | **0.807** | **0.961** |

## 7. Notes
- Image regeneration is deterministic — all seeds and prompts are stored in metadata.jsonl.
- Outliers are already excluded (is_outlier=False).
- All random splits use random_state=42 for reproducibility.
- Model outputs may vary slightly depending on library versions and hardware.

Author: Batuhan Cömert
Date: 23 November 2025
Institution: Boğaziçi University
