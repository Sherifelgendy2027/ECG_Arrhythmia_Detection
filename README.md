# 🫀 ECG Arrhythmia Detection

This repository contains an end-to-end pipeline for detecting cardiac arrhythmias from raw electrocardiogram (ECG) signals using the **MIT-BIH Arrhythmia Database**. 

## 📊 Dataset
This project uses the gold-standard benchmark in computational cardiology:
**[MIT-BIH Arrhythmia Database (Simple CSVs — Modern 2023)](https://www.kaggle.com/datasets/protobioengineering/mit-bih-arrhythmia-database-modern-2023)**

The dataset consists of 48 half-hour, two-channel ECG recordings sampled at 360 Hz. Cardiologists have annotated the exact timestamp (R-peak) and heartbeat type for every beat in the dataset.

## 🚀 Milestones

### ✅ Milestone 1: Exploratory Data Analysis & Classical ML Baseline
*See: [`milestone_1_eda_classical_ml.ipynb`](milestone_1_eda_classical_ml.ipynb)*

The foundational pipeline for heartbeat classification. The notebook maps all beats into binary labels: **Normal (N)** vs. **Abnormal (Any Arrhythmia)**.

#### Key Implementations:
1. **Automated Data Pathing:** Auto-detects whether running on Kaggle or locally. Includes instructions for downloading the Kaggle dataset locally using the Kaggle CLI.
2. **Signal Processing:** Implemented a 4th-order zero-phase Butterworth bandpass filter (0.5Hz – 45Hz) to remove baseline wander, power-line interference, and high-frequency muscle noise without shifting phase.
3. **Feature Engineering:** Tabularized the continuous ECG signal by extracting a fixed-duration waveform (-0.2s to +0.4s) around each annotated R-peak and computing time-domain statistics (mean, std, max, min, peak-to-peak, and RR-interval).
4. **Data Leakage Prevention:**
   * Used **Patient-Level Splitting** (`GroupShuffleSplit` by `record_id`) ensuring beats from the same patient *never* appear in both train and test sets.
   * Fitted the `StandardScaler` and `PCA` strictly on the training set to prevent statistical leakage.
### ✅ Milestone 2: Classical Machine Learning Suite & Deep Learning Benchmark
*See: [`machine_models.ipynb`](machine_models.ipynb)*

An advanced multi-model benchmark comparing 7 models (4 Classical ML vs. 3 Deep Learning architectures) on raw ECG waveforms and patient-normalized features.

#### Key Implementations:
1. **Patient-Normalized Features:** Z-scores amplitude features per record to eliminate inter-patient voltage variations while preserving intra-patient arrhythmia morphology.
2. **Classical ML Suite:** Trained and hyperparameter-tuned Logistic Regression, Random Forest, LightGBM, and Support Vector Machine (RBF kernel) using `RandomizedSearchCV` with 5-fold `GroupKFold`. SVM (RBF) is configurable to train on the full training set or a stratified subsample, trading runtime for tractability.
3. **Deep Learning Suite:** Trained 1D-CNN, CNN-BiLSTM, and CNN-BiLSTM with Attention architectures directly on 216-sample beat waveforms. (A fourth architecture, Hybrid ResNet + Tabular, was evaluated and removed after showing unstable/diverging validation loss and the weakest Macro F1 of any deep learning model at the highest training cost.)
4. **Overfitting/Underfitting Diagnostics:** Evaluated train vs. validation vs. test metrics across all classical and deep learning models.
5. **Statistical Significance & Thresholding:** Calibrated classical models' decision thresholds using $F_1$ and $F_{0.5}$ criteria via cross-validated out-of-fold (OOF) predictions across all training patients (5-fold `GroupKFold`) rather than a single small validation slice, for more stable calibration; deep learning thresholds are calibrated on the validation set. Conducted McNemar's test for pairwise statistical significance across top models.

## 🛠 Setup & Installation

To run this project locally, clone the repository and download the dataset:

```bash
# 1. Clone the repo
git clone https://github.com/Sherifelgendy2027/ECG_Arrhythmia_Detection.git
cd ECG_Arrhythmia_Detection

# 2. Download the dataset using Kaggle CLI (creates a ./data folder)
kaggle datasets download -d protobioengineering/mit-bih-arrhythmia-database-modern-2023 --path ./data --unzip
```

Alternatively, you can run the notebook directly on Kaggle by uploading the notebook and attaching the dataset.
