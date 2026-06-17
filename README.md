# Palm-Vein-Fusion
Another Fusion Model for Palm Vein Recognition.


This repository implements **decision-level fusion** of two complementary biometric classifiers for palm vein recognition, as described in the associated research paper. The code combines:

- **Decision I – BSIF + K‑NN**: Binarized Statistical Image Features (BSIF) extracted from five overlapping palm sub‑regions, concatenated into a histogram feature vector, and classified with a Manhattan‑distance *k*‑nearest neighbour (KNN) classifier.
- **Decision II – Vision Transformer (ViT)**: A `vit_small_patch16_224` model fine‑tuned end‑to‑end with strong data augmentation, label smoothing, AdamW optimizer and cosine learning rate schedule.
- **Fusion**: Weighted OR Rule (threshold = 0.9, paper §3.6). A test sample is accepted if the *sum of correctness weights* (1 if a decision is correct, 0 otherwise) reaches the threshold; otherwise it is rejected.

The pipeline runs on two public palm vein datasets: **PolyU** and **FYODB (FYO)**. Five repeated runs (seeds 11, 22, 33, 44, 55) are executed per dataset, each using a new random train/test split. All results are written live to a CSV file, with an “avg” summary row appended after all runs finish.

---

## Architecture

```
┌─────────────────────┐    ┌─────────────────────────────┐
│    Input Image      │    │     Input Image (BGR→RGB)    │
└────────┬────────────┘    └─────────────┬───────────────┘
         │                               │
         ▼                               ▼
  Grayscale + Resize              Strong Augmentations
         │                        (RandCrop, Flip, ColorJitter…)
         ▼                               │
   5 Overlapping                        ViT
   Sub‑regions                  vit_small_patch16_224
         │                               │
         ▼                               ▼
   BSIF encoding                  Decision II (ViT)
   (learned ICA filters)               
         │                               
         ▼                               
  Score‑level fusion                      
  (histogram concat)                     
         │                               
         ▼                               
  Manhattan KNN                           
         │                               
         ▼                               
 Decision I (BSIF+KNN)                    
         │                               
         └──────────┬────────────────────┘
                    │
                    ▼
            Weighted OR Rule
           (threshold = 0.9)
                    │
                    ▼
              Fused Decision
```

---

## Datasets

The code supports two datasets, configured via paths in the script. You can easily adapt it to your own directory structure.

| Dataset | Classes | Notes |
|---------|---------|-------|
| **PolyU** | 386 | Requires `ROI` folder containing `_roi.bmp` files. |
| **FYODB** | 160 | Two sessions: `Session1` and `Session2`. |

**Directory layout expected** (paths can be modified in `DATASET_CONFIGS` dictionary inside the script):

```
Data/
├── PolyUV2/
│   └── ROI/
│       ├── 001_1_roi.bmp
│       ├── 001_2_roi.bmp
│       └── ...
└── FYODB/
    └── FYODB/
        └── ROI/
            ├── Session1/
            │   ├── s001_1.png
            │   └── ...
            └── Session2/
                ├── s001_2.png
                └── ...
```

You can change the absolute paths by editing the `DATASET_CONFIGS` dictionary directly in the source code.

---

## Requirements

Install the following packages:

```bash
pip install torch torchvision timm scikit-learn opencv-python pillow scipy
```

- `timm` – Vision Transformer models
- `scipy` – loading `.mat` filter banks (if pre‑trained BSIF filters are provided)

Optionally, for GPU acceleration, install a CUDA‑compatible PyTorch build.

---

## Usage

```bash
python var-3.py [--datasets PolyU FYO] 
                      [--output-csv fusion_results.csv]
                      [--cache-dir _bsif_cache]
                      [--workers 4]
                      [--no-amp]
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--datasets` | `PolyU FYO` | List of datasets to process. Use `PolyU`, `FYO` (or `FYODB`). |
| `--output-csv` | `fusion_results.csv` (same folder as script) | Path for the output CSV file. |
| `--cache-dir` | `_bsif_cache` | Directory for BSIF filter and feature caches. Caches are automatically created and reused. |
| `--rebuild-cache` | `False` | Recompute BSIF filters and features even if a cache exists. |
| `--workers` | `4` | Number of DataLoader workers for ViT training (set to `0` on Windows if multiprocessing errors occur). |
| `--no-amp` | `False` | Disable automatic mixed precision (AMP). By default AMP is enabled if CUDA is available. |

---

## Output CSV Format

Every run appends one row to the CSV. After all 5 runs for a dataset, an `avg` row is appended.

| Column | Description |
|--------|-------------|
| `dataset` | Dataset alias (e.g., `PolyU`, `FYO`) |
| `run` | Run number (1‑5) or `avg` |
| `seed` | Random seed used for the run |
| `num_classes` | Number of classes in the dataset |
| `train_samples` / `test_samples` | Number of samples in train/test split |
| `d1_accuracy`, `d1_precision_macro`, ... | Decision I (BSIF+KNN) metrics |
| `d2_accuracy`, `d2_precision_macro`, ... | Decision II (ViT) metrics |
| `d2_best_epoch` | Epoch at which the best ViT accuracy was recorded |
| `fusion_accuracy`, `fusion_precision_macro`, ... | Fused decision metrics |
| `bsif_filter_build_time_sec` | Time to learn or load BSIF filters |
| `bsif_feature_build_time_sec` | Time to compute BSIF feature vectors |
| `d1_train_time_sec` | KNN training time |
| `d1_inference_time_sec` | KNN inference time on test split |
| `d2_train_time_sec` | ViT training time |
| `d2_inference_time_sec` | ViT inference time |
| `total_run_time_sec` | Total wall‑time for the run |
| `status` | `ok` if no errors, otherwise an error message |

---

## Reproducibility

- BSIF filters are learned via FastICA on random patches extracted from the training images. A cache (`.npz`) is saved per dataset and reused across the five random seeds unless `--rebuild-cache` is used.
- BSIF feature vectors are also cached to avoid recomputation.
- Each run uses a fixed seed (11, 22, 33, 44, 55) for data splitting, PyTorch and NumPy.
- The ViT is trained for 15 epochs with the hyper‑parameters listed in `VIT_COMBO` (same as the paper’s best configuration).

---

## Customization

### Using your own dataset

1. Add an entry to `DATASET_CONFIGS` with the necessary directory paths and `num_classes`.
2. Extend the `get_data()` function to load images and labels for your dataset.
3. Optionally provide a pre‑learned BSIF filter bank (`.npz` or `.mat`) in the cache directory; it will be automatically detected.

### Changing model hyper‑parameters

All important parameters are declared near the top of the script:

- **BSIF**: `TARGET_SIZE`, `SUBREGION_SIZE`, `BSIF_KERNEL_SIZE`, `BSIF_BITS`, `KNN_NEIGHBORS`
- **ViT**: `VIT_COMBO` dictionary (model name, learning rate, epochs, etc.)
- **Fusion**: `FUSION_THRESHOLD`
- **Experiment**: `RUN_SEEDS`, `TEST_SIZE`

Modify these constants before running.

---



## Authors

**Morteza Farrokhnejad**  \
**Prof. Dr. Hasan Demirel**  

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.