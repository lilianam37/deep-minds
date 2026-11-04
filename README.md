# Detecting Phishing Websites from the URL Alone

**COSC 50523: Deep Learning, Fall 2026 · Team Deep Minds**
Liliana Matte, Whitney Akred, Leiton Peterson

## The question

A phishing website imitates a legitimate one to steal passwords or payment details. We
classify a website as **phishing (1)** or **legitimate (0)**. Because missing a phishing site
is the more harmful mistake, **recall on the phishing class** is our main metric.

Our midterm showed that with all 50 of the dataset's numeric features, even logistic
regression is almost perfect (99.98% validation accuracy). Many of those features also need
the web page itself to be downloaded. So the final project asks a harder and more realistic
question:

> **When only the URL is available, does a neural network that reads the raw URL text
> outperform models built on hand-crafted URL features?**

## Dataset

**PhiUSIIL Phishing URL Dataset** (UCI Machine Learning Repository, CC BY 4.0):
https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

- 235,795 websites: 134,850 legitimate and 100,945 phishing (42.81% phishing)
- In the original file, `label = 1` means legitimate. We flip it so **phishing = 1**.
- The dataset is **not included** in this repo. Download the CSV and put it in your
  Google Drive folder (see Setup).

## Models

All models use the same rows and splits and are scored by the same function.

| Model | Input | Role |
|---|---|---|
| Logistic Regression | 22 hand-crafted URL features | Baseline |
| Random Forest | 22 hand-crafted URL features | Strong non-neural model |
| MLP (PyTorch) | 22 hand-crafted URL features | Neural network on features |
| Character-level CNN (PyTorch) | Raw URL text | Neural network that learns its own URL representation |

**How URLs become numbers.**
- *Feature models:* use the 22 numbers the dataset authors computed from each URL
  (length, digit and special-character ratios, number of subdomains, HTTPS, etc.).
- *CNN:*
  1. Each character seen in the **training** URLs gets an ID.
  2. Every URL becomes a list of IDs, cut or padded to 200 characters.
  3. An embedding layer learns a vector for each character.
  4. 1D convolutions (widths 3, 5, 7) find character patterns.

Architecture diagrams for all four models are in [`diagrams/`](diagrams/).

## Experimental protocol

- **Split:** 70 / 15 / 15 train / validation / test, stratified, `random_state = 42`
  (165,056 / 35,369 / 35,370 rows). These are the same rows as the midterm.
- **No leakage:**
  - The feature scaler and the CNN's character vocabulary are fit on the training rows only.
  - All tuning uses the validation split.
  - The test split is used **once**, by `week3/A4_final_test`.
- **Logging:** every run appends one row to `results_log.csv` with the model, the split,
  every hyperparameter and every metric (accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
  false positives, false negatives).
- **Hyperparameter sweeps:** LR (C), RF (number of trees × depth), MLP (size × dropout),
  CNN (kernel sizes, embedding size, filters, max URL length, one setting at a time).
- **Component Testing:**
  - Feature models without `URLSimilarityIndex`
  - CNN with `http(s)://` and `www.` stripped from the URL
  - CNN with no convolution layers
  - All models on a domain-grouped split, where no domain appears in both train and test
- **Seeds:** fixed at 42 for Python, NumPy, scikit-learn and PyTorch.

## Repository layout

```
deep-minds/
├── README.md
├── requirements.txt
├── shared/
│   ├── project_utils.py     # features list, data loading, URL encoding, evaluate_model, log_result
│   └── torch_models.py      # MLP, CharCNN, training loop with early stopping
├── notebooks/
│   ├── midterm_results/             # original 50-feature experiments (LR, RF, Keras MLP)
│   ├── week1/ … week4/      # final project notebooks, named by person (A/B/C)
├── diagrams/                # architecture diagrams (PDF + PNG)
└── results/                 # results_log.csv, final_test_results.csv, figures/
```

## Setup

1. Open the notebooks in **Google Colab**. For anything that trains the MLP or CNN, turn on
   the GPU: Runtime → Change runtime type → T4 GPU.
2. In Google Drive, create the folder `MyDrive/DeepMinds/` and put these in it:
   - `PhiUSIIL_Phishing_URL_Dataset.csv` (from the UCI link above)
   - `shared/project_utils.py` and `shared/torch_models.py` from this repo
3. Each notebook's first cell mounts Drive and adds that folder to the Python path.

## How to reproduce the results

Run the notebooks in this order. Each one reads the files the earlier ones saved.

| Step | Notebook | Output |
|---|---|---|
| 1 | `week1/A1_url_split` | `master_with_splits.csv` (URLs, labels, random + domain splits) |
| 2 | `week1/A2_logistic_regression_url` | LR baseline + weight plot |
| 3 | `week1/B1_random_forest_url` | RF + feature importance |
| 4 | `week1/B2_mlp_url_pytorch` | MLP + learning curve |
| 5 | `week1/C1_sweeps_lr_rf_mlp` | LR / RF / MLP sweeps (+ no-URLSimilarityIndex ablation) |
| 6 | `week2/C2_char_cnn` | Char-CNN, learning curve, saved model `char_cnn_default.pt` |
| 7 | `week2/A3_cnn_sweep` | CNN sweep |
| 8 | `week2/B3_sweep_plots` | Sweep figures |
| 9 | `week3/B4_ablations` | Prefix-strip, no-conv, domain-split ablations |
| 10 | `week3/A4_final_test` | **Test-set results** for the four final models |
| 11 | `week3/C3_error_analysis_tsne` | Misclassified URLs + t-SNE plot |
| 12 | `week4/B5_final_figures` | Report figures and LaTeX tables |
| 13 | `week4/C4_gradio_demo` | Paste-a-URL demo |

The midterm notebooks in `notebooks/midterm/` reproduce the earlier 50-feature results and
are not needed for the final results.

## Results

*To be filled in after `week3/A4_final_test` is run.*

| Model (test split) | Accuracy | Precision | Recall | F1 | Missed phishing | False alarms |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Random Forest | | | | | | |
| MLP | | | | | | |
| Char-CNN | | | | | | |

Final settings for each model: see the `final` rows in `results/results_log.csv`.

## Use of AI tools

We used Claude to help plan timing for the remaining work, and draft the architecture diagrams. The team ran every
experiment, debugged and verified the code, chose the final settings, and wrote all analysis
and conclusions.
