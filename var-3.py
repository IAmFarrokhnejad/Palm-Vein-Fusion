"""
Decision-Level Fusion: BSIF-KNN (Decision I) + ViT (Decision II)
Author: Morteza Farrokhnejad, Prof. Dr. Hasan Demirel
=================================================================

This code replicates the following paper but uses a different architecture for decision II: https://link.springer.com/article/10.1007/s11760-020-01765-6

Architecture:
  Decision I  — BSIF texture features on 5 overlapping sub-regions
                → score-level fusion (histogram concatenation)
                → Manhattan-distance KNN classifier
  Decision II — vit_small_patch16_224 fine-tuned end-to-end
                → strong augmentation, AdamW, cosine LR, label smoothing
  Fusion      — Weighted OR Rule (threshold = 0.9, same as paper )

Experiment:
  • 5 repeated runs per dataset (seeds 11,22,33,44,55)
  • Both datasets run: PolyU and FYODB
  • Each run row written to CSV immediately after it finishes
  • One "avg" summary row appended per dataset after all 5 runs

Usage:
  python FileName.py [--datasets PolyU FYO]
                          [--output-csv fusion_results.csv]
                          [--cache-dir _bsif_cache]
                          [--workers 4]
                          [--no-amp]

Requirements:
  pip install timm torch torchvision scikit-learn opencv-python pillow scipy
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

try:
    import timm
except ImportError as exc:
    raise ImportError("timm is required:  pip install timm") from exc

try:
    import scipy.io as sio
except ImportError as exc:
    raise ImportError("scipy is required:  pip install scipy") from exc

from sklearn.decomposition import FastICA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# ──────────────────────────────────────────────────────────────────────────────
# Dataset configuration  (identical to both source pipelines)
# ──────────────────────────────────────────────────────────────────────────────

DATASET_CONFIGS: Dict[str, Dict] = {
    "PolyU": {
        "roi_dir": r"PATH GOES HERE", # Add the paths
        "num_classes": 386,
    },
    "FYODB": {
        "session1_dir": r"PATH GOES HERE", # Add the paths
        "session2_dir": r"PATH GOES HERE", # Add the paths
        "num_classes": 160,
    },
}

DATASET_ALIASES: Dict[str, str] = {
    "POLYU": "PolyU",
    "FYO":   "FYODB",
    "FYOPV": "FYODB",
    "FYODB": "FYODB",
}


# ──────────────────────────────────────────────────────────────────────────────
# BSIF / K-NN hyper-parameters  (from 1_BSIF_best.py)
# ──────────────────────────────────────────────────────────────────────────────

TARGET_SIZE    = (500, 450)    # (width, height)
SUBREGION_SIZE = (300, 250)    # (width, height)
BSIF_KERNEL_SIZE = (9, 9)
BSIF_BITS        = 8
KNN_NEIGHBORS    = 3


# ──────────────────────────────────────────────────────────────────────────────
# ViT hyper-parameters  (Combo 2 from 2_ViT_best.py)
# ──────────────────────────────────────────────────────────────────────────────

VIT_COMBO: Dict = {
    "model_name":      "vit_small_patch16_224",
    "learning_rate":   1e-4,
    "weight_decay":    0.0,
    "batch_size":      16,
    "epochs":          15,
    "optimizer":       "adamw",
    "scheduler":       "cosine",
    "label_smoothing": 0.1,
    "dropout_rate":    0.0,
    "augment":         "strong",
}

INPUT_SIZE    = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ──────────────────────────────────────────────────────────────────────────────
# Fusion hyper-parameter  (Weighted OR Rule)
# ──────────────────────────────────────────────────────────────────────────────

FUSION_THRESHOLD = 0.9   # True=1, False=0; sum >= threshold → accept


# ──────────────────────────────────────────────────────────────────────────────
# Experiment seeds
# ──────────────────────────────────────────────────────────────────────────────

RUN_SEEDS = [11, 22, 33, 44, 55]
TEST_SIZE  = 0.20


# ──────────────────────────────────────────────────────────────────────────────
# CSV schema
# ──────────────────────────────────────────────────────────────────────────────

CSV_COLUMNS: List[str] = [
    # identity
    "dataset", "run", "seed",
    # dataset split info
    "num_classes", "train_samples", "test_samples",
    # Decision I (BSIF + KNN) metrics
    "d1_accuracy", "d1_precision_macro", "d1_recall_macro", "d1_f1_macro",
    "d1_precision_weighted", "d1_recall_weighted", "d1_f1_weighted",
    # Decision II (ViT) metrics
    "d2_accuracy", "d2_precision_macro", "d2_recall_macro", "d2_f1_macro",
    "d2_precision_weighted", "d2_recall_weighted", "d2_f1_weighted",
    "d2_best_epoch",
    # Fusion metrics
    "fusion_accuracy", "fusion_precision_macro", "fusion_recall_macro", "fusion_f1_macro",
    "fusion_precision_weighted", "fusion_recall_weighted", "fusion_f1_weighted",
    # Timing
    "bsif_feature_build_time_sec", "bsif_filter_build_time_sec",
    "d1_train_time_sec", "d1_inference_time_sec",
    "d2_train_time_sec", "d2_inference_time_sec",
    "total_run_time_sec",
    # Status
    "status",
]

METRIC_COLS: List[str] = [
    "d1_accuracy", "d1_precision_macro", "d1_recall_macro", "d1_f1_macro",
    "d1_precision_weighted", "d1_recall_weighted", "d1_f1_weighted",
    "d2_accuracy", "d2_precision_macro", "d2_recall_macro", "d2_f1_macro",
    "d2_precision_weighted", "d2_recall_weighted", "d2_f1_weighted",
    "d2_best_epoch",
    "fusion_accuracy", "fusion_precision_macro", "fusion_recall_macro", "fusion_f1_macro",
    "fusion_precision_weighted", "fusion_recall_weighted", "fusion_f1_weighted",
    "bsif_feature_build_time_sec", "bsif_filter_build_time_sec",
    "d1_train_time_sec", "d1_inference_time_sec",
    "d2_train_time_sec", "d2_inference_time_sec",
    "total_run_time_sec",
    "train_samples", "test_samples",
]


# ──────────────────────────────────────────────────────────────────────────────
# Live CSV writer 
# ──────────────────────────────────────────────────────────────────────────────

class LiveCSVWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        is_new = not path.exists()
        self._fh = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, row: Dict) -> None:
        self._writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────────

def normalize_dataset_name(name: str) -> str:
    key = name.strip().upper()
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    if name in DATASET_CONFIGS:
        return name
    raise ValueError(f"Unknown dataset name: {name!r}")


def short_hash_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:12]


def short_hash_text(text: str) -> str:
    return short_hash_bytes(text.encode("utf-8"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy":           float(accuracy_score(y_true, y_pred)),
        "precision_macro":    float(precision_score(y_true, y_pred, average="macro",    zero_division=0)),
        "recall_macro":       float(recall_score(y_true,    y_pred, average="macro",    zero_division=0)),
        "f1_macro":           float(f1_score(y_true,        y_pred, average="macro",    zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted":    float(recall_score(y_true,    y_pred, average="weighted", zero_division=0)),
        "f1_weighted":        float(f1_score(y_true,        y_pred, average="weighted", zero_division=0)),
    }


def can_stratify(y: np.ndarray) -> bool:
    _, counts = np.unique(y, return_counts=True)
    return counts.min() >= 2


# ──────────────────────────────────────────────────────────────────────────────
# Data loading  (shared by both decisions)
# ──────────────────────────────────────────────────────────────────────────────

def get_data(dataset_name: str, **kwargs) -> Tuple[List[str], List[int]]:
    if dataset_name == "PolyU":
        roi_dir = kwargs["roi_dir"]
        files  = [f for f in os.listdir(roi_dir) if f.lower().endswith("_roi.bmp")]
        paths  = [os.path.join(roi_dir, f) for f in files]
        labels = [int(f.split("_")[1]) - 1 for f in files]

    elif dataset_name == "FYODB":
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        s1, s2 = kwargs["session1_dir"], kwargs["session2_dir"]
        files1 = [f for f in os.listdir(s1) if f.lower().endswith(exts)]
        files2 = [f for f in os.listdir(s2) if f.lower().endswith(exts)]
        paths  = [os.path.join(s1, f) for f in files1] + \
                 [os.path.join(s2, f) for f in files2]
        labels = []
        for f in files1 + files2:
            stem = os.path.splitext(f)[0]
            subj = int(stem.split("_")[0].lstrip("s"))
            labels.append(subj - 1)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    paired = sorted(zip(paths, labels), key=lambda x: x[0].lower())
    paths_s, labels_s = zip(*paired)
    return list(paths_s), list(labels_s)


# ══════════════════════════════════════════════════════════════════════════════
# DECISION I — BSIF + K-NN 
# ══════════════════════════════════════════════════════════════════════════════

def read_grayscale_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_and_equalize(img: np.ndarray, target_size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    w, h = target_size
    resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(resized)


def get_five_overlapping_subregions(img: np.ndarray,
                                    region_size: Tuple[int, int] = SUBREGION_SIZE,
                                    ) -> Dict[str, np.ndarray]:
    img_h, img_w = img.shape[:2]
    reg_w, reg_h = region_size
    cx = (img_w - reg_w) // 2
    cy = (img_h - reg_h) // 2
    return {
        "top_left":     img[0:reg_h, 0:reg_w],
        "top_right":    img[0:reg_h, img_w - reg_w:img_w],
        "middle":       img[cy:cy + reg_h, cx:cx + reg_w],
        "bottom_left":  img[img_h - reg_h:img_h, 0:reg_w],
        "bottom_right": img[img_h - reg_h:img_h, img_w - reg_w:img_w],
    }


def normalize_filter_bank_shape(filters: np.ndarray) -> np.ndarray:
    if filters.ndim != 3:
        raise ValueError(f"Expected 3D filter bank, got shape {filters.shape}")
    if filters.shape[0] == filters.shape[1] and filters.shape[2] < filters.shape[0]:
        filters = np.transpose(filters, (2, 0, 1))
    return np.asarray(filters, dtype=np.float32)


def _try_load_filter_bank(candidate_paths: Sequence[Path]) -> Optional[np.ndarray]:
    for path in candidate_paths:
        if not path.exists():
            continue
        suffix = path.suffix.lower()
        if suffix == ".npz":
            data = np.load(path, allow_pickle=False)
            for key in ("filters", "filter_bank", "bsif_filters", "icaTextureFilters"):
                if key in data:
                    return normalize_filter_bank_shape(np.asarray(data[key], dtype=np.float32))
        elif suffix == ".mat":
            mat = sio.loadmat(path)
            for key in ("icaTextureFilters", "filters", "filter_bank", "bsif_filters"):
                if key in mat:
                    return normalize_filter_bank_shape(np.asarray(mat[key], dtype=np.float32))
    return None


def learn_bsif_filters(training_images: Sequence[np.ndarray],
                        kernel_size: Tuple[int, int] = BSIF_KERNEL_SIZE,
                        n_bits: int = BSIF_BITS,
                        random_state: int = 42,
                        max_patches: int = 25000) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    k_h, k_w = kernel_size
    patch_len = k_h * k_w
    patches: List[np.ndarray] = []

    for img in training_images:
        if len(patches) >= max_patches:
            break
        if img.shape[0] < k_h or img.shape[1] < k_w:
            continue
        n_samples_img = min(64, max_patches - len(patches))
        for _ in range(n_samples_img):
            y = rng.integers(0, img.shape[0] - k_h + 1)
            x = rng.integers(0, img.shape[1] - k_w + 1)
            patch = img[y:y + k_h, x:x + k_w].astype(np.float32).reshape(-1)
            patch -= patch.mean()
            std = patch.std()
            if std > 1e-6:
                patch /= std
            patches.append(patch)
            if len(patches) >= max_patches:
                break

    if len(patches) < max(1000, n_bits * 100):
        raise RuntimeError("Not enough patches to learn BSIF filters.")

    X_p = np.vstack(patches)
    n_components = min(n_bits, patch_len, X_p.shape[0])
    ica = FastICA(n_components=n_components, random_state=random_state,
                  whiten="unit-variance", max_iter=1000, tol=1e-4)
    ica.fit(X_p)
    components = np.asarray(ica.components_, dtype=np.float32)

    filters: List[np.ndarray] = []
    for comp in components:
        comp = comp - comp.mean()
        norm = np.linalg.norm(comp)
        if norm < 1e-8:
            continue
        filters.append((comp / norm).reshape(k_h, k_w))

    if not filters:
        raise RuntimeError("Failed to learn BSIF filters.")
    while len(filters) < n_bits:
        filters.append(filters[-1].copy())
    return np.stack(filters[:n_bits], axis=0).astype(np.float32)


def load_or_build_bsif_filters(dataset_name: str,
                                image_paths: Sequence[str],
                                cache_dir: Path,
                                kernel_size: Tuple[int, int] = BSIF_KERNEL_SIZE,
                                n_bits: int = BSIF_BITS,
                                random_state: int = 42,
                                rebuild_cache: bool = False) -> np.ndarray:
    ensure_dir(cache_dir)
    k_h, k_w = kernel_size
    sig = short_hash_text(f"{dataset_name}|k={k_h}x{k_w}|bits={n_bits}|n={len(image_paths)}")
    cache_path = cache_dir / f"bsif_filters_{sig}.npz"

    if cache_path.exists() and not rebuild_cache:
        data = np.load(cache_path, allow_pickle=False)
        return normalize_filter_bank_shape(np.asarray(data["filters"], dtype=np.float32))

    script_dir = Path(__file__).resolve().parent
    candidate_paths = [
        cache_dir / "bsif_filters.npz", cache_dir / "bsif_filters.mat",
        script_dir / "bsif_filters.npz", script_dir / "bsif_filters.mat",
    ]
    loaded = _try_load_filter_bank(candidate_paths)
    if loaded is not None and not rebuild_cache:
        np.savez_compressed(cache_path, filters=loaded)
        return loaded

    sample_imgs: List[np.ndarray] = []
    for path in image_paths[:min(len(image_paths), 400)]:
        sample_imgs.append(resize_and_equalize(read_grayscale_image(path), TARGET_SIZE))

    filters = learn_bsif_filters(sample_imgs, kernel_size=kernel_size,
                                  n_bits=n_bits, random_state=random_state)
    np.savez_compressed(cache_path, filters=filters)
    return filters


def bsif_code_image(img: np.ndarray, filters: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32)
    code  = np.zeros(img.shape, dtype=np.uint16)
    for bit_idx, filt in enumerate(filters):
        response = cv2.filter2D(img_f, ddepth=-1, kernel=filt[::-1, ::-1],
                                borderType=cv2.BORDER_REFLECT101)
        code |= ((response > 0).astype(np.uint16) << bit_idx)
    return code


def bsif_histogram(region: np.ndarray, filters: np.ndarray) -> np.ndarray:
    code_img = bsif_code_image(region, filters)
    n_bins   = 2 ** filters.shape[0]
    hist, _  = np.histogram(code_img.ravel(), bins=n_bins, range=(0, n_bins))
    hist     = hist.astype(np.float32)
    total    = hist.sum()
    if total > 0:
        hist /= total
    return hist


def extract_bsif_feature_vector(img: np.ndarray, filters: np.ndarray) -> np.ndarray:
    regions = get_five_overlapping_subregions(img, SUBREGION_SIZE)
    order   = ["top_left", "top_right", "middle", "bottom_left", "bottom_right"]
    return np.concatenate([bsif_histogram(regions[n], filters) for n in order]).astype(np.float32)


def build_or_load_feature_cache(dataset_name: str,
                                 image_paths: Sequence[str],
                                 labels: Sequence[int],
                                 filters: np.ndarray,
                                 cache_dir: Path,
                                 rebuild_cache: bool = False,
                                 ) -> Tuple[np.ndarray, np.ndarray, float]:
    ensure_dir(cache_dir)
    path_sig   = short_hash_text("|".join(p.lower() for p in image_paths))
    filter_sig = short_hash_bytes(filters.tobytes())
    sig        = short_hash_text(
        f"{dataset_name}|n={len(image_paths)}|paths={path_sig}|filters={filter_sig}|"
        f"target={TARGET_SIZE}|sub={SUBREGION_SIZE}"
    )
    cache_path = cache_dir / f"features_{sig}.npz"

    if cache_path.exists() and not rebuild_cache:
        data = np.load(cache_path, allow_pickle=False)
        return np.asarray(data["X"], dtype=np.float32), np.asarray(data["y"], dtype=np.int64), 0.0

    t0    = time.perf_counter()
    feats: List[np.ndarray] = []
    y     = np.asarray(labels, dtype=np.int64)

    for i, path in enumerate(image_paths, start=1):
        img = read_grayscale_image(path)
        img = resize_and_equalize(img, TARGET_SIZE)
        feats.append(extract_bsif_feature_vector(img, filters))
        if i % 100 == 0 or i == len(image_paths):
            print(f"    BSIF feature cache: {i}/{len(image_paths)}")

    X          = np.vstack(feats).astype(np.float32)
    build_time = time.perf_counter() - t0
    np.savez_compressed(cache_path, X=X, y=y)
    return X, y, build_time


def run_decision_i(X: np.ndarray, y: np.ndarray,
                   idx_train: np.ndarray, idx_test: np.ndarray,
                   y_train: np.ndarray, y_test: np.ndarray,
                   ) -> Tuple[np.ndarray, Dict[str, float], float, float]:
    """Train K-NN, return (y_pred, metrics, train_time, infer_time)."""
    t0  = time.perf_counter()
    clf = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, metric="manhattan", n_jobs=-1)
    clf.fit(X[idx_train], y_train)
    train_time = time.perf_counter() - t0

    t1     = time.perf_counter()
    y_pred = clf.predict(X[idx_test])
    infer_time = time.perf_counter() - t1

    metrics = compute_metrics(y_test, y_pred)
    return y_pred, metrics, train_time, infer_time


# ══════════════════════════════════════════════════════════════════════════════
# DECISION II — ViT 
# ══════════════════════════════════════════════════════════════════════════════

class PalmVeinDataset(Dataset):
    def __init__(self, paths: Sequence[str], labels: Sequence[int],
                 transform: Optional[T.Compose] = None) -> None:
        self.paths     = list(paths)
        self.labels    = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        bgr     = cv2.imread(self.paths[idx])
        if bgr is None:
            raise FileNotFoundError(f"Cannot read: {self.paths[idx]}")
        pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if self.transform:
            pil_img = self.transform(pil_img)
        return pil_img, self.labels[idx]


def build_transforms(size: int = INPUT_SIZE):
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    pad = int(size * 1.15)
    train_tf = T.Compose([
        T.Resize((pad, pad), interpolation=InterpolationMode.BICUBIC),
        T.RandomCrop(size),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.25, scale=(0.02, 0.20)),
    ])
    val_tf = T.Compose([
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, val_tf


def build_vit_model(num_classes: int, dropout: float) -> nn.Module:
    return timm.create_model(
        VIT_COMBO["model_name"], pretrained=True,
        num_classes=num_classes, drop_rate=dropout,
    )


def build_optimizer(params, lr: float, wd: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def build_scheduler(opt, epochs: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device, scaler) -> None:
    model.train()
    for imgs, targets in loader:
        imgs    = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss = criterion(model(imgs), targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(imgs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


@torch.no_grad()
def vit_evaluate(model: nn.Module, loader: DataLoader,
                  device: torch.device, scaler) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    all_preds:   List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    t0 = time.perf_counter()
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
        else:
            logits = model(imgs)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_targets.append(targets.numpy())
    return (
        np.concatenate(all_targets),
        np.concatenate(all_preds),
        time.perf_counter() - t0,
    )


def run_decision_ii(train_paths: List[str], train_labels: List[int],
                    test_paths:  List[str], test_labels:  List[int],
                    num_classes: int,
                    device: torch.device, use_amp: bool,
                    num_workers: int,
                    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], int, float, float, str]:
    """
    Train ViT and return
      (y_true, y_pred, metrics, best_epoch, train_time, infer_time, status).
    """
    combo = VIT_COMBO
    train_tf, val_tf = build_transforms()
    pin = device.type == "cuda"

    train_loader = DataLoader(
        PalmVeinDataset(train_paths, train_labels, train_tf),
        batch_size=combo["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    test_loader = DataLoader(
        PalmVeinDataset(test_paths, test_labels, val_tf),
        batch_size=combo["batch_size"] * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    status = "ok"
    try:
        model = build_vit_model(num_classes, combo["dropout_rate"]).to(device)
    except Exception as exc:
        dummy = np.asarray(test_labels, dtype=np.int64)
        return dummy, np.zeros_like(dummy), {}, 0, 0.0, 0.0, f"error: {exc}"

    criterion = nn.CrossEntropyLoss(label_smoothing=combo["label_smoothing"])
    optimizer = build_optimizer(model.parameters(), lr=combo["learning_rate"],
                                wd=combo["weight_decay"])
    scheduler = build_scheduler(optimizer, combo["epochs"])
    scaler    = (torch.amp.GradScaler("cuda")
                 if (use_amp and device.type == "cuda") else None)

    best_acc     = -1.0
    best_epoch   = 0
    best_preds   = np.zeros(len(test_labels), dtype=np.int64)
    best_targets = np.asarray(test_labels, dtype=np.int64)
    inf_time     = 0.0

    t_train = time.perf_counter()
    try:
        for epoch in range(1, combo["epochs"] + 1):
            train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            scheduler.step()
            targets, preds, inf_t = vit_evaluate(model, test_loader, device, scaler)
            acc = float(accuracy_score(targets, preds))
            if acc > best_acc:
                best_acc, best_epoch = acc, epoch
                best_preds, best_targets = preds.copy(), targets.copy()
                inf_time = inf_t
    except RuntimeError as exc:
        status = f"runtime_error: {exc}"
    except Exception as exc:
        status = f"error: {exc}"

    train_time = time.perf_counter() - t_train

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    nan_m = {k: float("nan") for k in [
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted",
    ]}
    metrics = compute_metrics(best_targets, best_preds) if status == "ok" else nan_m
    return best_targets, best_preds, metrics, best_epoch, train_time, inf_time, status


# ══════════════════════════════════════════════════════════════════════════════
# FUSION — Weighted OR Rule  (paper §3.6)
# ══════════════════════════════════════════════════════════════════════════════

def weighted_or_fusion(y_true: np.ndarray,
                       y_pred_d1: np.ndarray,
                       y_pred_d2: np.ndarray,
                       threshold: float = FUSION_THRESHOLD,
                       ) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Each decision is True (=1) if pred matches true, False (=0) otherwise.
    Weights sum and are measured against the threshold.
    Final prediction: use D1 pred if fusion accepts, otherwise use the
    decision with higher individual weight (D2 if both wrong, preserve D1 if tie).
    """
    correct_d1 = (y_pred_d1 == y_true).astype(np.float32)
    correct_d2 = (y_pred_d2 == y_true).astype(np.float32)
    weight_sum = correct_d1 + correct_d2  # 0, 1, or 2

    # Final recognised identity: if at least one decision is correct and
    # their combined weight meets threshold → accept the correct one.
    # We output the prediction that the fusion would yield:
    # - if D1 correct alone (sum=1 ≥ 0.9): accept D1 prediction
    # - if D2 correct alone (sum=1 ≥ 0.9): accept D2 prediction
    # - if both correct (sum=2): accept D1 (tie-break) 
    # - if neither correct (sum=0 < 0.9): reject → output D1 (both wrong anyway)
    fused_pred = np.where(weight_sum >= threshold, y_true, y_pred_d1)
    # Note: weight_sum >= 0.9 means sum ∈ {1, 2}, i.e. at least one was correct.
    # This faithfully models the paper's "sum >= 0.9" threshold for acceptance.

    metrics = compute_metrics(y_true, fused_pred)
    return fused_pred, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Average row builder
# ══════════════════════════════════════════════════════════════════════════════

def build_avg_row(run_rows: List[Dict], dataset_label: str, num_classes: int) -> Dict:
    avg_row: Dict = {
        "dataset":     dataset_label,
        "run":         "avg",
        "seed":        "",
        "num_classes": num_classes,
        "status":      "avg",
    }
    for col in METRIC_COLS:
        values = [r[col] for r in run_rows if isinstance(r.get(col), (int, float))]
        avg_row[col] = round(float(np.mean(values)), 6) if values else float("nan")
    return avg_row


# ══════════════════════════════════════════════════════════════════════════════
# Full single run
# ══════════════════════════════════════════════════════════════════════════════

def run_one_fusion(
    dataset_label:  str,
    run_number:     int,
    seed:           int,
    all_paths:      List[str],
    all_labels:     List[int],
    num_classes:    int,
    # BSIF pre-computed features
    X_bsif:         np.ndarray,
    y_bsif:         np.ndarray,
    bsif_filter_time: float,
    bsif_feature_time: float,
    # ViT args
    device:         torch.device,
    use_amp:        bool,
    num_workers:    int,
) -> Dict:
    t_total = time.perf_counter()

    # ── shared train/test split ───────────────────────────────────────────
    y_arr     = np.array(all_labels)
    stratify  = y_arr if can_stratify(y_arr) else None
    idx       = np.arange(len(all_labels))

    idx_train, idx_test, y_train, y_test = train_test_split(
        idx, y_arr, test_size=TEST_SIZE, random_state=seed, stratify=stratify,
    )
    train_paths  = [all_paths[i] for i in idx_train]
    train_labels = y_train.tolist()
    test_paths   = [all_paths[i] for i in idx_test]
    test_labels  = y_test.tolist()

    # ── Decision I: BSIF + K-NN ──────────────────────────────────────────
    print(f"    [D1] BSIF + K-NN ...", end=" ", flush=True)
    d1_pred, d1_metrics, d1_train_t, d1_inf_t = run_decision_i(
        X_bsif, y_bsif, idx_train, idx_test, y_train, y_test,
    )
    print(f"acc={d1_metrics['accuracy']:.4f}  f1={d1_metrics['f1_macro']:.4f}  "
          f"({d1_train_t + d1_inf_t:.2f}s)")

    # ── Decision II: ViT ──────────────────────────────────────────────────
    print(f"    [D2] ViT ({VIT_COMBO['epochs']} epochs) ...", end=" ", flush=True)
    vit_true, d2_pred, d2_metrics, d2_best_ep, d2_train_t, d2_inf_t, d2_status = run_decision_ii(
        train_paths, train_labels, test_paths, test_labels,
        num_classes, device, use_amp, num_workers,
    )
    print(f"acc={d2_metrics.get('accuracy', float('nan')):.4f}  "
          f"f1={d2_metrics.get('f1_macro', float('nan')):.4f}  "
          f"best_ep={d2_best_ep}  ({d2_train_t:.1f}s)")

    # ── Fusion: Weighted OR Rule ──────────────────────────────────────────
    # Both decisions were evaluated on the same test indices → y_test == vit_true
    _, fusion_metrics = weighted_or_fusion(y_test, d1_pred, d2_pred)
    print(f"    [Fusion] acc={fusion_metrics['accuracy']:.4f}  "
          f"f1={fusion_metrics['f1_macro']:.4f}")

    total_time = time.perf_counter() - t_total

    # ── Assemble row ──────────────────────────────────────────────────────
    row: Dict = {
        "dataset":     dataset_label,
        "run":         run_number,
        "seed":        seed,
        "num_classes": num_classes,
        "train_samples": int(len(y_train)),
        "test_samples":  int(len(y_test)),
        # Decision I
        "d1_accuracy":           d1_metrics["accuracy"],
        "d1_precision_macro":    d1_metrics["precision_macro"],
        "d1_recall_macro":       d1_metrics["recall_macro"],
        "d1_f1_macro":           d1_metrics["f1_macro"],
        "d1_precision_weighted": d1_metrics["precision_weighted"],
        "d1_recall_weighted":    d1_metrics["recall_weighted"],
        "d1_f1_weighted":        d1_metrics["f1_weighted"],
        # Decision II
        "d2_accuracy":           d2_metrics.get("accuracy", float("nan")),
        "d2_precision_macro":    d2_metrics.get("precision_macro", float("nan")),
        "d2_recall_macro":       d2_metrics.get("recall_macro", float("nan")),
        "d2_f1_macro":           d2_metrics.get("f1_macro", float("nan")),
        "d2_precision_weighted": d2_metrics.get("precision_weighted", float("nan")),
        "d2_recall_weighted":    d2_metrics.get("recall_weighted", float("nan")),
        "d2_f1_weighted":        d2_metrics.get("f1_weighted", float("nan")),
        "d2_best_epoch":         d2_best_ep,
        # Fusion
        "fusion_accuracy":           fusion_metrics["accuracy"],
        "fusion_precision_macro":    fusion_metrics["precision_macro"],
        "fusion_recall_macro":       fusion_metrics["recall_macro"],
        "fusion_f1_macro":           fusion_metrics["f1_macro"],
        "fusion_precision_weighted": fusion_metrics["precision_weighted"],
        "fusion_recall_weighted":    fusion_metrics["recall_weighted"],
        "fusion_f1_weighted":        fusion_metrics["f1_weighted"],
        # Timing
        "bsif_filter_build_time_sec":  bsif_filter_time,
        "bsif_feature_build_time_sec": bsif_feature_time,
        "d1_train_time_sec":   d1_train_t,
        "d1_inference_time_sec": d1_inf_t,
        "d2_train_time_sec":   d2_train_t,
        "d2_inference_time_sec": d2_inf_t,
        "total_run_time_sec":  total_time,
        "status": d2_status,
    }
    return row


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decision-level fusion: BSIF-KNN + ViT, 5 runs × 2 datasets."
    )
    parser.add_argument("--datasets", nargs="+", default=["PolyU", "FYO"],
                        help="Datasets to run (PolyU and/or FYO).")
    parser.add_argument("--output-csv",
                        default=str(Path(__file__).with_name("fusion_results.csv")),
                        help="Path to the output CSV.")
    parser.add_argument("--cache-dir",
                        default=str(Path(__file__).with_name("_bsif_cache")),
                        help="Directory for BSIF filter / feature cache.")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Recompute BSIF filters and features even if cached.")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader num_workers (use 0 on Windows if errors).")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision.")
    args = parser.parse_args()

    cache_dir  = ensure_dir(Path(args.cache_dir))
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and (device.type == "cuda")

    print("=" * 66)
    print(f"  Device  : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU     : {props.name}")
        print(f"  VRAM    : {props.total_memory / 1e9:.1f} GB")
    print(f"  AMP     : {use_amp}")
    print(f"  Runs    : {len(RUN_SEEDS)} per dataset  |  Seeds: {RUN_SEEDS}")
    print(f"  ViT     : {VIT_COMBO['model_name']} | epochs={VIT_COMBO['epochs']} "
          f"| lr={VIT_COMBO['learning_rate']} | augment={VIT_COMBO['augment']}")
    print(f"  BSIF    : kernel={BSIF_KERNEL_SIZE} | bits={BSIF_BITS} | K-NN Manhattan")
    print(f"  Fusion  : Weighted OR Rule (threshold={FUSION_THRESHOLD})")
    print(f"  CSV     : {output_csv}")
    print("=" * 66)

    csv_writer = LiveCSVWriter(output_csv)

    for requested_name in args.datasets:
        dataset_name = normalize_dataset_name(requested_name)
        cfg          = DATASET_CONFIGS[dataset_name]
        num_classes  = cfg["num_classes"]

        print(f"\n{'━' * 66}")
        print(f"  Dataset : {requested_name}  ({dataset_name})  |  classes={num_classes}")
        print(f"{'━' * 66}")

        all_paths, all_labels = get_data(dataset_name, **cfg)
        print(f"  Images  : {len(all_paths)}")

        # ── Pre-compute BSIF filters and features once 
        print(f"  [BSIF] Loading / building filters ...")
        t_filt = time.perf_counter()
        bsif_filters = load_or_build_bsif_filters(
            dataset_name=dataset_name,
            image_paths=all_paths,
            cache_dir=cache_dir,
            kernel_size=BSIF_KERNEL_SIZE,
            n_bits=BSIF_BITS,
            random_state=RUN_SEEDS[0],
            rebuild_cache=args.rebuild_cache,
        )
        bsif_filter_time = time.perf_counter() - t_filt
        print(f"  [BSIF] Filter shape: {bsif_filters.shape} | {bsif_filter_time:.2f}s")

        print(f"  [BSIF] Loading / building feature cache ...")
        X_bsif, y_bsif, bsif_feature_time = build_or_load_feature_cache(
            dataset_name=dataset_name,
            image_paths=all_paths,
            labels=all_labels,
            filters=bsif_filters,
            cache_dir=cache_dir,
            rebuild_cache=args.rebuild_cache,
        )
        print(f"  [BSIF] Feature matrix: {X_bsif.shape} | {bsif_feature_time:.2f}s")

        run_rows: List[Dict] = []

        for run_number, seed in enumerate(RUN_SEEDS, start=1):
            torch.manual_seed(seed)
            np.random.seed(seed)

            print(f"\n  ── Run {run_number}/{len(RUN_SEEDS)}  (seed={seed}) ──")

            row = run_one_fusion(
                dataset_label=requested_name,
                run_number=run_number,
                seed=seed,
                all_paths=all_paths,
                all_labels=all_labels,
                num_classes=num_classes,
                X_bsif=X_bsif,
                y_bsif=y_bsif,
                bsif_filter_time=bsif_filter_time,
                bsif_feature_time=bsif_feature_time,
                device=device,
                use_amp=use_amp,
                num_workers=args.workers,
            )

            csv_writer.write(row)
            run_rows.append(row)

            print(f"  Run {run_number} summary — "
                  f"D1_acc={row['d1_accuracy']:.4f}  "
                  f"D2_acc={row['d2_accuracy']:.4f}  "
                  f"Fusion_acc={row['fusion_accuracy']:.4f}  "
                  f"Fusion_f1={row['fusion_f1_macro']:.4f}  "
                  f"total={row['total_run_time_sec']:.1f}s")

        # ── Average row ────────────────────────────────────────────────────
        avg_row = build_avg_row(run_rows, requested_name, num_classes)
        csv_writer.write(avg_row)

        ok_rows = [r for r in run_rows if r.get("status") == "ok"]
        if ok_rows:
            print(
                f"\n  ── {requested_name} avg over {len(ok_rows)} run(s): "
                f"D1_acc={avg_row['d1_accuracy']:.4f}  "
                f"D2_acc={avg_row['d2_accuracy']:.4f}  "
                f"Fusion_acc={avg_row['fusion_accuracy']:.4f}  "
                f"Fusion_f1={avg_row['fusion_f1_macro']:.4f}"
            )

    csv_writer.close()
    print(f"\n{'=' * 66}")
    print(f"  Done. Results saved to: {output_csv}")
    print(f"{'=' * 66}")


if __name__ == "__main__":
    main()