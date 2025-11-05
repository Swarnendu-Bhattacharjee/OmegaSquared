# 🧠 Omega² — Temporal Credit Risk Modeling Framework

**Authors:** Chitro Majumdar, Sergio Scandizzo, Ratan Mahanta, Avradip Mandal and Swarnendu Bhattacharjee

---

## 📘 Introduction

**Omega²** is a temporal credit risk modeling framework designed to produce forward-looking, interpretable, and reproducible corporate credit scores. The framework integrates structured financial data with advanced ensemble learning and temporal validation schemes to ensure real-world reliability.

This repository provides only the **Python scripts** and **LaTeX/PNG files** necessary to reproduce the paper's results. No raw or processed datasets are included, in accordance with publication and confidentiality requirements.

**Keywords:** Corporate Credit Scoring, Temporal Validation, Machine Learning, Financial Risk Analytics, XGBoost, Regression, Classification

---

## 📁 Repository Structure

```
OmegaSquared/
│
├── src/
│   ├── filter_by_agency.py
│   ├── filteredpromax_kaggle_for_folders.py
│   ├── filteredpromax_kaggle_dedup_for_folders.py
│   ├── filteredpromax_kaggle_with_year_month_for_folders.py
│   ├── run_temporal_classification_all_agencies_fixed.py
│   ├── run_temporal_regression_all_agencies_fixed.py
│   └── run_temporal_xgb_groupsplit_all_agencies.py
│
├── paper/
│   ├── Omega^2.tex
│   ├── classification_results_table_corrected.tex
│   ├── regression_results_table_corrected.tex
│   └── media/
│       ├── *.png
│       └── ...
│
├── README.md
├── LICENSE
├── .gitignore
└── requirements.txt
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Swarnendu-Bhattacharjee/OmegaSquared.git
cd OmegaSquared
```

### 2️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Kaggle Dataset Setup

The Kaggle dataset used in this project must be downloaded manually due to licensing restrictions.

**Steps:**

1. Download the dataset from Kaggle (referenced in the paper).
2. Create a local directory named `data/` inside the project root.
3. Place all raw Kaggle files in:

   ```bash
   OmegaSquared/data/raw/
   ```
4. The processing scripts will automatically create and store processed CSV files in:

   ```bash
   OmegaSquared/data/processed/
   ```

---

## 🚀 Running the Workflow

### Step 1 — Data Preparation

```bash
python src/filteredpromax_kaggle_for_folders.py
python src/filteredpromax_kaggle_dedup_for_folders.py
python src/filteredpromax_kaggle_with_year_month_for_folders.py
python src/filter_by_agency.py
```

### Step 2 — Model Execution

```bash
python src/run_temporal_classification_all_agencies_fixed.py
python src/run_temporal_regression_all_agencies_fixed.py
python src/run_temporal_xgb_groupsplit_all_agencies.py
```

These scripts will reproduce all figures and metrics reported in the paper.

---

## 📊 Expected Outputs

* Classification AUC: **> 0.93**
* Regression R²: **> 0.60**
* RMSE: **Sub-milliscale**
* Temporal Consistency: **Cross-year AUC Std < 0.02**

Generated outputs (figures, plots, and tables) will be stored under `paper/media/`.

---

## 🧠 Technologies Used

* **Python 3.10+**
* **Scikit-learn**, **XGBoost**, **LightGBM**, **CatBoost**
* **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
* **Optuna** (Bayesian hyperparameter optimization)
* **LaTeX** (for paper compilation)

---

## 🪪 License

This repository is released under the **CC BY-NC-SA License**.
See the [LICENSE](LICENSE) file for details.

---

© 2025 RsRL
