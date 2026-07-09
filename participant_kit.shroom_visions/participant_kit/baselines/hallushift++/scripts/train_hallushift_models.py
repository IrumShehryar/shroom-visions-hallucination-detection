#!/usr/bin/env python3
"""Train HalluShift classifiers for all models on the original per-token JSONs.

This script reads the non-redacted `hidden/per_token_hidden_*_{model}` folders,
builds token-level features from raw per-token vectors, and trains a 3-layer
MLP for each model.

Supports two label modes:
- binary: hallucinated vs none
- multiclass: predict the hallucination type, including `none`

For each model it prints a random-guess baseline and the trained model metrics.
It also writes a CSV summary to `hidden/hallushift_model_results.csv` by default.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


NONE_LABEL = "none"
DEFAULT_MODELS = ["llava", "internvl", "minicpm", "qwenvl", "gemma3"]

SCALAR_FEATURE_COLUMNS = [
    "max_prob",
    "perplexity",
    "inverse_max_prob",
    "target_prob",
    "token_nll",
    "target_token_perplexity",
    "layer_consistency",
    "layer_inconsistency",
    "attn_gini_mean_last3",
    "attn_gini_std_last3",
    "hspp_mean_inverse_max_prob",
    "hspp_std_inverse_max_prob",
    "hspp_confidence_trend",
    "hspp_mean_confidence",
    "hspp_low_conf_frac",
    "unique_repetition_ratio",
    "bigram_repetition_ratio",
    "normalized_unique_tokens",
    "token_text_len",
    "token_index",
    "response_len",
]


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def extract_spans(label_entries: object) -> List[Dict[str, object]]:
    spans: List[Dict[str, object]] = []
    if not isinstance(label_entries, list):
        return spans

    for entry in label_entries:
        payload = None
        label_hint = None

        if isinstance(entry, dict):
            payload = entry
        elif isinstance(entry, (list, tuple)):
            for item in entry:
                if isinstance(item, dict):
                    payload = item
                    break
            if entry and not isinstance(entry[0], dict):
                label_hint = entry[0]

        if not isinstance(payload, dict):
            continue

        start = payload.get("start")
        end = payload.get("end")
        if start is None or end is None:
            continue

        label = (
            payload.get("label")
            or payload.get("hallucination")
            or payload.get("hallucination_label")
            or payload.get("type")
            or payload.get("category")
            or payload.get("tag")
            or label_hint
            or "hallucination"
        )

        spans.append({"start": int(start), "end": int(end), "label": str(label)})

    return spans


def build_lookup_from_jsonl(jsonl_path: Path) -> Dict[Tuple[str, str, str], List[Dict[str, object]]]:
    lookup: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            image_name = None
            for key in ["image", "image_name", "image_path", "img", "filename", "file"]:
                if key in obj and obj[key]:
                    image_name = Path(str(obj[key])).name
                    break

            prompt = None
            for key in ["prompt", "question", "q"]:
                if key in obj and obj[key]:
                    prompt = str(obj[key])
                    break

            response = None
            for key in ["response", "answer", "a"]:
                if key in obj and obj[key]:
                    response = str(obj[key])
                    break

            if image_name is None or prompt is None or response is None:
                continue
            lookup[(image_name, prompt, response)] = extract_spans(obj.get("labels", []))

    return lookup


def find_token_span(response_text: str, token_text: str, cursor: int) -> Tuple[int, int, int]:
    if not token_text:
        return -1, -1, cursor

    idx = response_text.find(token_text, cursor)
    if idx >= 0:
        return idx, idx + len(token_text), idx + len(token_text)

    cleaned = token_text.strip("▁")
    if cleaned:
        idx = response_text.find(cleaned, cursor)
        if idx >= 0:
            return idx, idx + len(cleaned), idx + len(cleaned)

    return -1, -1, cursor


def to_fixed(values: object, dim: int) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    if isinstance(values, list) and dim > 0:
        vals = np.array(values[:dim], dtype=np.float32)
        out[: len(vals)] = vals
    return out


def detect_project_root() -> Path:
    here = Path(__file__).resolve().parent
    candidate = here.parent
    if (candidate / "hidden").exists():
        return candidate
    return Path.cwd().resolve()


def discover_per_token_files(hidden_dir: Path, model_name: str) -> List[Path]:
    # Support model_name variants like "llava" or "llava_redacted".
    # If the caller requests a "_redacted" variant, only pick files with the
    # redacted filename suffix. Otherwise pick the canonical per-token files.
    if model_name.endswith("_redacted"):
        base = model_name[: -len("_redacted")]
        pattern = f"**/per_token_hidden_*_*_{base}/all_samples_per_token_{base}*_redacted.json"
        return sorted(hidden_dir.glob(pattern))

    base = model_name
    pattern = f"**/per_token_hidden_*_*_{base}/all_samples_per_token_{base}*.json"
    files = sorted(hidden_dir.glob(pattern))
    # Exclude files that are the redacted variants when running the normal model
    non_redacted = [p for p in files if not p.name.endswith("_redacted.json")]
    return non_redacted


def parse_split_from_folder(folder_name: str, model_name: str) -> Optional[Tuple[str, str]]:
    # Allow callers to pass model_name or model_name + '_redacted'.
    match_model = model_name
    if model_name.endswith("_redacted"):
        match_model = model_name[: -len("_redacted")]
    match = re.match(rf"per_token_hidden_([a-z]+)_(train|test)_{re.escape(match_model)}$", folder_name)
    if not match:
        return None
    return match.group(1), match.group(2)


@dataclass
class TokenRow:
    language: str
    split: str
    sentence_id: str
    token_index: int
    token_text: str
    max_prob: float
    perplexity: float
    hidden_layer_norms: List[float]
    attn_means: List[float]
    attn_entropies: List[float]
    response_len: int
    label_name: str


def build_rows_for_model(project_root: Path, hidden_dir: Path, model_name: str) -> pd.DataFrame:
    per_token_files = discover_per_token_files(hidden_dir, model_name)
    if not per_token_files:
        print(f"  No per-token files found for {model_name}")
        return pd.DataFrame()
    
    print(f"  Found {len(per_token_files)} per-token file(s) for {model_name}")

    jsonl_cache: Dict[str, Dict[Tuple[str, str, str], List[Dict[str, object]]]] = {}
    records: List[dict] = []

    for file_path in per_token_files:
        split_info = parse_split_from_folder(file_path.parent.name, model_name)
        if split_info is None:
            continue
        language, split_name = split_info

        with open(file_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        metadata = payload.get("metadata", {})
        samples = payload.get("samples", [])
        jsonl_source = metadata.get("jsonl_source")
        if not jsonl_source:
            continue

        jsonl_path = project_root / jsonl_source
        if not jsonl_path.exists():
            continue

        cache_key = str(jsonl_path)
        if cache_key not in jsonl_cache:
            jsonl_cache[cache_key] = build_lookup_from_jsonl(jsonl_path)
        label_lookup = jsonl_cache[cache_key]

        print(f"    Processing {len(samples)} samples from {language} {split_name}")
        for sample_idx, sample in enumerate(samples):
            meta = sample.get("meta", {})
            image = Path(str(meta.get("image", ""))).name
            prompt = str(meta.get("prompt", ""))
            response = str(meta.get("response", ""))
            spans = label_lookup.get((image, prompt, response), [])

            sentence_id = f"{language}:{split_name}:{sample_idx}:{image}"
            cursor = 0

            for tok in sample.get("tokens", []):
                token_text = str(tok.get("token_text") or tok.get("raw_token") or "")
                start, end, cursor = find_token_span(response, token_text, cursor)

                label_name = NONE_LABEL
                if start >= 0:
                    for span in spans:
                        if end <= span["start"] or start >= span["end"]:
                            continue
                        label_name = str(span["label"])
                        break

                records.append(
                    {
                        "language": language,
                        "split": split_name,
                        "sentence_id": sentence_id,
                        "token_index": int(tok.get("token_index", 0)),
                        "token_text": token_text,
                        "max_prob": safe_float(tok.get("max_prob")),
                        "perplexity": safe_float(tok.get("perplexity")),
                        "inverse_max_prob": safe_float(tok.get("inverse_max_prob", tok.get("perplexity"))),
                        "target_prob": safe_float(tok.get("target_prob")),
                        "token_nll": safe_float(tok.get("token_nll")),
                        "target_token_perplexity": safe_float(tok.get("target_token_perplexity")),
                        "layer_consistency": safe_float(tok.get("layer_consistency")),
                        "layer_inconsistency": safe_float(tok.get("layer_inconsistency")),
                        "attn_gini_mean_last3": safe_float(tok.get("attn_gini_mean_last3")),
                        "attn_gini_std_last3": safe_float(tok.get("attn_gini_std_last3")),
                        "hspp_mean_inverse_max_prob": safe_float(tok.get("hspp_mean_inverse_max_prob")),
                        "hspp_std_inverse_max_prob": safe_float(tok.get("hspp_std_inverse_max_prob")),
                        "hspp_confidence_trend": safe_float(tok.get("hspp_confidence_trend")),
                        "hspp_mean_confidence": safe_float(tok.get("hspp_mean_confidence")),
                        "hspp_low_conf_frac": safe_float(tok.get("hspp_low_conf_frac")),
                        "unique_repetition_ratio": safe_float(tok.get("unique_repetition_ratio")),
                        "bigram_repetition_ratio": safe_float(tok.get("bigram_repetition_ratio")),
                        "normalized_unique_tokens": safe_float(tok.get("normalized_unique_tokens")),
                        "token_text_len": float(len(token_text)),
                        "hidden_layer_norms": tok.get("hidden_layer_norms", []),
                        "attn_means": tok.get("attn_means", []),
                        "attn_entropies": tok.get("attn_entropies", []),
                        "response_len": len(response),
                        "label_name": label_name,
                    }
                )

    if not records:
        print(f"  No records created for {model_name}")
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    print(f"  Built {len(records)} total token records for {model_name}")
    print(f"    - Unique sentences: {df['sentence_id'].nunique()}")
    print(f"    - Languages: {df['language'].unique().tolist()}")
    print(f"    - Train/Test split: {(df['split'] == 'train').sum()}/{(df['split'] == 'test').sum()}")
    print(f"    - Label distribution: {dict(df['label_name'].value_counts())}")
    return df


def make_labels(df: pd.DataFrame, label_mode: str) -> Tuple[np.ndarray, Dict[str, int]]:
    if label_mode == "binary":
        y = df["label_name"].fillna(NONE_LABEL).apply(lambda v: 0 if v == NONE_LABEL else 1).astype(np.int64).values
        return y, {"none": 0, "hallucination": 1}

    label_names = sorted(set(df["label_name"].fillna(NONE_LABEL).astype(str).tolist()))
    if NONE_LABEL not in label_names:
        label_names.insert(0, NONE_LABEL)
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    y = df["label_name"].fillna(NONE_LABEL).astype(str).map(label_to_id).astype(np.int64).values
    return y, label_to_id


def features_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, int, int, int]:
    hidden_norms_list = df["hidden_layer_norms"].tolist()
    attn_means_list = df["attn_means"].tolist()
    attn_entropies_list = df["attn_entropies"].tolist()

    max_hidden_dim = max((len(v) for v in hidden_norms_list if isinstance(v, list)), default=0)
    max_attn_mean_dim = max((len(v) for v in attn_means_list if isinstance(v, list)), default=0)
    max_attn_entropy_dim = max((len(v) for v in attn_entropies_list if isinstance(v, list)), default=0)

    scalar_columns = [c for c in SCALAR_FEATURE_COLUMNS if c in df.columns]

    print(f"  Feature dimensions:")
    print(f"    - Hidden layer norms: {max_hidden_dim}")
    print(f"    - Attention means: {max_attn_mean_dim}")
    print(f"    - Attention entropies: {max_attn_entropy_dim}")
    print(f"    - Scalar features: {len(scalar_columns)}")
    print(f"    - Scalar names: {scalar_columns}")
    total_dim = max_hidden_dim + max_attn_mean_dim + max_attn_entropy_dim + len(scalar_columns)
    print(f"    - Total feature dimension: {total_dim}")

    rows = []
    for idx, row in df.iterrows():
        hn = to_fixed(hidden_norms_list[idx], max_hidden_dim)
        am = to_fixed(attn_means_list[idx], max_attn_mean_dim)
        ae = to_fixed(attn_entropies_list[idx], max_attn_entropy_dim)
        scalars = np.array([safe_float(row.get(c, 0.0)) for c in scalar_columns], dtype=np.float32)
        rows.append(np.concatenate([hn, am, ae, scalars], axis=0))

    X = np.stack(rows).astype(np.float32)
    print(f"  Feature matrix shape: {X.shape}")
    return X, max_hidden_dim, max_attn_mean_dim, max_attn_entropy_dim


class TokenDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str, label_mode: str) -> Tuple[float, float, float, float, float]:
    model.eval()
    preds: List[int] = []
    trues: List[int] = []
    losses: List[float] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(loss.item())
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            trues.extend(yb.cpu().numpy().tolist())

    if label_mode == "binary":
        prec, rec, f1, _ = precision_recall_fscore_support(trues, preds, average="macro", zero_division=0)
    else:
        prec, rec, f1, _ = precision_recall_fscore_support(trues, preds, average="macro", zero_division=0)
    acc = accuracy_score(trues, preds)
    return float(np.mean(losses)) if losses else 0.0, float(acc), float(prec), float(rec), float(f1)


def random_baseline(y_true: np.ndarray, y_train: np.ndarray, label_mode: str, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    classes = np.unique(y_train)
    probs = np.bincount(y_train, minlength=int(classes.max()) + 1).astype(np.float64)
    probs = probs / probs.sum()

    preds = rng.choice(np.arange(len(probs)), size=len(y_true), p=probs)
    acc = accuracy_score(y_true, preds)
    if label_mode == "binary":
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    else:
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="macro", zero_division=0)
    return {"acc": float(acc), "prec": float(prec), "rec": float(rec), "f1": float(f1)}

def evaluate_sklearn(clf, X, y, label_mode: str):
    y_pred = clf.predict(X)

    avg = "binary" if label_mode == "binary" else "macro"

    prec, rec, f1, _ = precision_recall_fscore_support(
        y,
        y_pred,
        average=avg,
        zero_division=0,
    )

    acc = accuracy_score(y, y_pred)

    return {
        "acc": float(acc),
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
    }

def train_one_model(
    project_root: Path,
    hidden_dir: Path,
    model_name: str,
    train_on_redacted: bool,
    label_mode: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_sentence_ratio: float,
    seed: int,
    device: str,
    max_train_tokens: Optional[int] = None,
    max_test_tokens: Optional[int] = None,
    quick_mode: bool = False,
    model_config: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    
    if model_config is None:
        model_config = {"classifier": "mlp"}

    classifier_name = str(model_config.get("classifier", "mlp"))

    print(f"\n[DATA LOADING] Building data for {model_name}...")
    train_source_model = f"{model_name}_redacted" if train_on_redacted else model_name
    test_source_model = model_name
    print(f"  Train source model: {train_source_model}")
    print(f"  Test source model: {test_source_model}")

    train_source_df = build_rows_for_model(project_root, hidden_dir, train_source_model)
    if train_source_df.empty:
        print(f"  ERROR: No training data found")
        return {"model": model_name, "status": "no_train_data"}

    if train_source_model == test_source_model:
        test_source_df = train_source_df
    else:
        test_source_df = build_rows_for_model(project_root, hidden_dir, test_source_model)
        if test_source_df.empty:
            print(f"  ERROR: No test data found")
            return {"model": model_name, "status": "no_test_data"}

    train_df = train_source_df[train_source_df["split"] == "train"].copy()
    test_df = test_source_df[test_source_df["split"] == "test"].copy()
    if train_df.empty or test_df.empty:
        print(f"  ERROR: Missing train or test split")
        return {"model": model_name, "status": "missing_train_or_test"}
    
    print(f"  Train tokens: {len(train_df)}, Test tokens: {len(test_df)}")

    if quick_mode:
        print(f"\n[QUICK MODE] Limiting token counts...")
        np.random.seed(seed)
        if max_train_tokens is not None:
            train_sids = train_df["sentence_id"].unique()
            np.random.shuffle(train_sids)
            selected_train_sids = []
            running = 0
            sentence_sizes = train_df.groupby("sentence_id").size().to_dict()
            for sid in train_sids:
                selected_train_sids.append(sid)
                running += int(sentence_sizes[sid])
                if running >= max_train_tokens:
                    break
            original_train_size = len(train_df)
            train_df = train_df[train_df["sentence_id"].isin(selected_train_sids)].copy()
            print(f"  Train: {original_train_size} -> {len(train_df)} tokens")

        if max_test_tokens is not None:
            test_sids = test_df["sentence_id"].unique()
            np.random.shuffle(test_sids)
            selected_test_sids = []
            running = 0
            sentence_sizes = test_df.groupby("sentence_id").size().to_dict()
            for sid in test_sids:
                selected_test_sids.append(sid)
                running += int(sentence_sizes[sid])
                if running >= max_test_tokens:
                    break
            original_test_size = len(test_df)
            test_df = test_df[test_df["sentence_id"].isin(selected_test_sids)].copy()
            print(f"  Test: {original_test_size} -> {len(test_df)} tokens")

    print(f"\n[FEATURE EXTRACTION] Building features...")
    model_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    y, label_to_id = make_labels(model_df, label_mode)
    print(f"  Labels: {label_to_id}")
    X, max_hidden_dim, max_attn_mean_dim, max_attn_entropy_dim = features_from_df(model_df)

    print(f"\n[DATA SPLITTING] Creating train/val/test splits...")
    train_sentence_ids = train_df["sentence_id"].unique()
    train_sids, val_sids = train_test_split(train_sentence_ids, test_size=val_sentence_ratio, random_state=seed)

    train_mask = model_df["sentence_id"].isin(train_sids).values
    val_mask = model_df["sentence_id"].isin(val_sids).values
    test_mask = model_df["split"].eq("test").values

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"  Train: {len(X_train)} tokens ({len(train_sids)} sentences)")
    print(f"  Val: {len(X_val)} tokens ({len(val_sids)} sentences)")
    print(f"  Test: {len(X_test)} tokens")
    print(f"  Train label distribution: {dict(pd.Series(y_train).value_counts())}")
    print(f"  Val label distribution: {dict(pd.Series(y_val).value_counts())}")
    print(f"  Test label distribution: {dict(pd.Series(y_test).value_counts())}")

    train_ds = TokenDataset(X_train, y_train)
    val_ds = TokenDataset(X_val, y_val)
    test_ds = TokenDataset(X_test, y_test)

    class_counts = np.bincount(y_train, minlength=len(label_to_id) if label_mode == "multiclass" else 2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[y_train]
    train_sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"\n[MODEL TRAINING] Building classifier: {classifier_name}")
    num_classes = 2 if label_mode == "binary" else len(label_to_id)

    best_epoch = -1
    best_val_f1 = -1.0
    test_loss = 0.0

    if classifier_name == "mlp":
        model = MLP(X.shape[1], num_classes).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(model_config.get("lr", lr)),
            weight_decay=float(model_config.get("weight_decay", 0.0)),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(model_config.get("epochs", epochs)),
        )

        criterion = nn.CrossEntropyLoss()
        best_state = None

        def clone_state_dict(state_dict):
            return {k: v.detach().cpu().clone() for k, v in state_dict.items()}

        run_epochs = int(model_config.get("epochs", epochs))
        run_batch_size = int(model_config.get("batch_size", batch_size))

        train_loader = DataLoader(train_ds, batch_size=run_batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_ds, batch_size=run_batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=run_batch_size, shuffle=False)

        for epoch in range(1, run_epochs + 1):
            model.train()
            train_losses = []

            for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{run_epochs}"):
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            scheduler.step()

            val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
                model,
                val_loader,
                criterion,
                device,
                label_mode,
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_state = clone_state_dict(model.state_dict())

            print(
                f"  Epoch {epoch}/{run_epochs}: "
                f"train_loss={np.mean(train_losses):.4f} "
                f"val_f1={val_f1:.4f}"
            )

        if best_state is None:
            best_state = clone_state_dict(model.state_dict())

        best_model = MLP(X.shape[1], num_classes).to(device)
        best_model.load_state_dict(best_state)

        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
            best_model,
            test_loader,
            criterion,
            device,
            label_mode,
        )

    elif classifier_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=int(model_config.get("n_estimators", 300)),
            max_depth=model_config.get("max_depth", None),
            min_samples_leaf=int(model_config.get("min_samples_leaf", 1)),
            class_weight=model_config.get("class_weight", "balanced"),
            random_state=seed,
            n_jobs=int(model_config.get("n_jobs", -1)),
        )

        clf.fit(X_train, y_train)

        val_metrics = evaluate_sklearn(clf, X_val, y_val, label_mode)
        test_metrics = evaluate_sklearn(clf, X_test, y_test, label_mode)

        best_val_f1 = val_metrics["f1"]
        test_acc = test_metrics["acc"]
        test_prec = test_metrics["prec"]
        test_rec = test_metrics["rec"]
        test_f1 = test_metrics["f1"]
        best_model = clf

    elif classifier_name == "logistic_regression":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(model_config.get("C", 1.0)),
                class_weight=model_config.get("class_weight", "balanced"),
                max_iter=int(model_config.get("max_iter", 2000)),
                random_state=seed,
            ),
        )

        clf.fit(X_train, y_train)

        val_metrics = evaluate_sklearn(clf, X_val, y_val, label_mode)
        test_metrics = evaluate_sklearn(clf, X_test, y_test, label_mode)

        best_val_f1 = val_metrics["f1"]
        test_acc = test_metrics["acc"]
        test_prec = test_metrics["prec"]
        test_rec = test_metrics["rec"]
        test_f1 = test_metrics["f1"]
        best_model = clf
    elif classifier_name == "svm":
        clf = make_pipeline(
            StandardScaler(),
            SVC(
                C=float(model_config.get("C", 1.0)),
                gamma=model_config.get("gamma", "scale"),
                kernel=model_config.get("kernel", "rbf"),
                class_weight=model_config.get("class_weight", "balanced"),
                random_state=seed,
            ),
        )

        clf.fit(X_train, y_train)

        val_metrics = evaluate_sklearn(clf, X_val, y_val, label_mode)
        test_metrics = evaluate_sklearn(clf, X_test, y_test, label_mode)

        best_val_f1 = val_metrics["f1"]
        test_acc = test_metrics["acc"]
        test_prec = test_metrics["prec"]
        test_rec = test_metrics["rec"]
        test_f1 = test_metrics["f1"]
        best_model = clf
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Test metrics: acc={test_acc:.4f} prec={test_prec:.4f} rec={test_rec:.4f} f1={test_f1:.4f}")
    
    print(f"  Computing random baseline...")
    baseline = random_baseline(y_test, y_train, label_mode, seed)
    print(f"  Baseline metrics: acc={baseline['acc']:.4f} prec={baseline['prec']:.4f} rec={baseline['rec']:.4f} f1={baseline['f1']:.4f}")

    print(f"\n[SUMMARY] Model: {model_name}")
    improvement = test_f1 - baseline["f1"]
    print(f"  F1 improvement over baseline: {improvement:.4f} ({test_f1:.4f} vs {baseline['f1']:.4f})")
    
    results = {
        "classifier": classifier_name,
        "model_config": json.dumps(model_config),
        "model": model_name,
        "train_source_model": train_source_model,
        "test_source_model": test_source_model,
        "label_mode": label_mode,
        "status": "ok",
        "best_epoch": best_epoch,
        "train_tokens": int(len(X_train)),
        "val_tokens": int(len(X_val)),
        "test_tokens": int(len(X_test)),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_prec": float(test_prec),
        "test_rec": float(test_rec),
        "test_f1": float(test_f1),
        "random_acc": baseline["acc"],
        "random_prec": baseline["prec"],
        "random_rec": baseline["rec"],
        "random_f1": baseline["f1"],
        "best_val_f1": float(best_val_f1),
        "num_classes": int(num_classes),
        "input_dim": int(X.shape[1]),
        "max_hidden_dim": int(max_hidden_dim),
        "max_attn_mean_dim": int(max_attn_mean_dim),
        "max_attn_entropy_dim": int(max_attn_entropy_dim),
        "label_to_id": label_to_id,
    }
    # --- Export per-token predictions for the test set ---
    try:
        print("\n[EXPORT] Generating per-token prediction outputs...")
        # Build a dataframe for the test rows in the same order as X_test
        df_test = model_df[test_mask].reset_index(drop=True)

        all_preds = []

        if classifier_name == "mlp":
            best_model.eval()

            with torch.no_grad():
                for xb, _ in test_loader:
                    xb = xb.to(device)
                    logits = best_model(xb)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    preds = np.argmax(probs, axis=1)

                    all_preds.extend(preds.tolist())
                    all_probs.extend(probs.tolist())

        else:
            preds = clf.predict(X_test)

            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_test)
            else:
                # fallback for models without predict_proba
                probs = np.zeros((len(preds), len(label_to_id)), dtype=float)
                probs[np.arange(len(preds)), preds] = 1.0

            all_preds = preds.tolist()
            all_probs = probs.tolist()

        id_to_label = {v: k for k, v in label_to_id.items()}
        rows = []
        for i in range(len(df_test)):
            true_label = df_test.loc[i, "label_name"]
            pred_id = int(all_preds[i]) if i < len(all_preds) else -1
            pred_label = id_to_label.get(pred_id, "UNKNOWN")
            prob_vec = all_probs[i] if i < len(all_probs) else []
            pred_prob = float(prob_vec[pred_id]) if (isinstance(prob_vec, (list, np.ndarray)) and pred_id >= 0 and pred_id < len(prob_vec)) else None

            rows.append(
                {
                    "classifier": classifier_name,
                    "model_config": json.dumps(model_config),
                    "model": model_name,
                    "label_mode": label_mode,
                    "language": df_test.loc[i, "language"],
                    "split": df_test.loc[i, "split"],
                    "sentence_id": df_test.loc[i, "sentence_id"],
                    "token_index": int(df_test.loc[i, "token_index"]),
                    "token_text": df_test.loc[i, "token_text"],
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "pred_prob": pred_prob,
                    "pred_probs": json.dumps(prob_vec),
                }
            )

        preds_df = pd.DataFrame(rows)
        train_tag = "_train_redacted" if train_on_redacted else ""
        classifier_tag = safe_filename(model_config.get("classifier", "mlp"))

        run_tag = model_config.get("run_name")
        if run_tag is None:
            run_tag = model_config.get("trial_id")

        if run_tag is None:
            run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        run_tag = safe_filename(run_tag)

        preds_csv = (
            project_root
            / "hidden"
            / "predictions"
            / f"hallushift_predictions_{model_name}{train_tag}_{label_mode}_{classifier_tag}_{run_tag}.csv"
        )
        preds_csv.parent.mkdir(parents=True, exist_ok=True)
        preds_df.to_csv(preds_csv, index=False)
        print(f"  ✓ Saved predictions to: {preds_csv}")
    except Exception as e:
        print(f"  ✗ Failed to export predictions: {e}")

    return results

def safe_filename(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)
    return text.strip("_")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HalluShift models on original per-token JSONs.")
    parser.add_argument("--models", default="all", help="Comma-separated model names or 'all'.")
    parser.add_argument(
        "--train-on-redacted",
        action="store_true",
        help="Use redacted per-token JSONs for the training split, but keep the full files for test evaluation.",
    )
    parser.add_argument("--label-mode", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-sentence-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick-mode", action="store_true", help="Limit the number of tokens for faster runs.")
    parser.add_argument("--max-train-tokens", type=int, default=None)
    parser.add_argument("--max-test-tokens", type=int, default=None)
    parser.add_argument("--results-csv", default=None, help="Optional results CSV path. Defaults to a label-mode specific file in hidden/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    project_root = detect_project_root()
    hidden_dir = project_root / "hidden"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.models == "all":
        models = DEFAULT_MODELS
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    print("Project root:", project_root)
    print("Hidden dir:", hidden_dir)
    print("Device:", device)
    print("Models:", models)
    print("Train on redacted:", args.train_on_redacted)
    print("Label mode:", args.label_mode)
    print("Quick mode:", args.quick_mode)

    results = []
    for idx, model_name in enumerate(models, 1):
        print("\n" + "=" * 100)
        print(f"[{idx}/{len(models)}] Training Model: {model_name.upper()}")
        print("=" * 100)
        result = train_one_model(
            project_root=project_root,
            hidden_dir=hidden_dir,
            model_name=model_name,
            train_on_redacted=args.train_on_redacted,
            label_mode=args.label_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_sentence_ratio=args.val_sentence_ratio,
            seed=args.seed,
            device=device,
            max_train_tokens=args.max_train_tokens,
            max_test_tokens=args.max_test_tokens,
            quick_mode=args.quick_mode,
        )
        results.append(result)

        if result.get("status") != "ok":
            print(f"  ✗ SKIPPED: {result.get('status')}")
            continue

        print(f"  ✓ COMPLETED SUCCESSFULLY")

    print("\n" + "=" * 100)
    print("[FINAL RESULTS]")
    print("=" * 100)
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(by=["test_f1", "test_acc"], ascending=False).reset_index(drop=True)
        
        # Display key metrics
        print(f"\nProcessed {len(results_df)} models in {args.label_mode} mode")
        print(f"\nResults sorted by F1 score:")
        print(results_df[["model", "test_f1", "test_acc", "test_prec", "test_rec", "best_epoch", "test_tokens"]].to_string(index=False))
        
        print(f"\nDetailed results:")
        print(results_df.to_string(index=False))

        if args.results_csv is None:
            results_csv = project_root / "hidden" / f"hallushift_model_results_{args.label_mode}.csv"
        else:
            results_csv = project_root / args.results_csv
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_csv, index=False)
        print(f"\n✓ Saved results to: {results_csv}")
    else:
        print("\n✗ No results to save")


if __name__ == "__main__":
    main()
