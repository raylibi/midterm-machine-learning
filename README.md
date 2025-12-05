# Machine Learning Midterm Projects

**Name:** Rayhan Diff  
**NIM:** 1103220039  
**Class:** [Insert Your Class Here]

---

## Repository Purpose
This repository serves as a submission for the Machine Learning Midterm Exam. It contains three distinct machine learning projects demonstrating proficiency in **Classification**, **Regression**, and **Clustering**. 

The projects utilize advanced techniques including Gradient Boosting (XGBoost) and GPU-accelerated libraries (RAPIDS) to solve real-world problems ranging from financial fraud detection to customer segmentation.

---

## Repository Navigation & Project Overview

This repository consists of three main Jupyter Notebooks, each focusing on a specific machine learning domain:

| File Name | Domain | Project Title | Key Libraries |
| :--- | :--- | :--- | :--- |
| `MidtermML-1.ipynb` | **Classification** | Online Fraud Detection | XGBoost, Pandas, Scikit-Learn |
| `MidtermML-2.ipynb` | **Regression** | Song Release Year Prediction | XGBoost, Seaborn |
| `MidtermML-3.ipynb` | **Clustering** | Credit Card Customer Segmentation | RAPIDS (cuML, cuDF), PCA |

---

## Project Details & Results

### 1. Fraud Detection (Classification)
**File:** `MidtermML-1.ipynb`

* **Objective:** To detect fraudulent online transactions within a highly imbalanced dataset.
* **Methodology:**
    * Data Preprocessing: Label encoding and handling missing values.
    * Handling Imbalance: Used `scale_pos_weight` to address the ~1:27 fraud-to-legit ratio.
    * Model: **XGBoost Classifier** (GPU accelerated).
* **Key Results:**
    * The model achieved high discriminative ability with a focus on Recall to ensure fraud cases are captured.
    
    | Metric | Score |
    | :--- | :--- |
    | **ROC-AUC Score** | **0.9664** |
    | **Precision (Fraud)** | 0.48 |
    | **Recall (Fraud)** | 0.82 |
    | **F1-Score (Weighted)** | 0.97 |

### 2. Audio Feature Analysis (Regression)
**File:** `MidtermML-2.ipynb`

* **Objective:** To predict the release year of songs based on audio features (timbre, frequency, etc.) using the Million Song Dataset.
* **Methodology:**
    * Data Scaling: Standardization using `StandardScaler`.
    * Model: **XGBoost Regressor** (`tree_method='hist'` for GPU efficiency).
* **Key Results:**
    * The model predicts the song release year with an average error margin of approximately 6 years (MAE).
    
    | Metric | Score |
    | :--- | :--- |
    | **RMSE** | 8.5773 |
    | **MAE** | 5.9781 |
    | **R² Score** | 0.3818 |

### 3. Customer Segmentation (Clustering)
**File:** `MidtermML-3.ipynb`

* **Objective:** To segment credit card users into distinct behavioral groups to aid marketing strategies.
* **Methodology:**
    * **Tech Stack:** Used **RAPIDS (cuDF, cuML)** for high-performance GPU clustering.
    * Technique: **K-Means Clustering** with dimensionality reduction via **PCA** for visualization.
    * Optimization: Used the Elbow Method to determine the optimal number of clusters (K=4).
* **Key Results:**
    * **Silhouette Score:** 0.2481
    * **Identified Segments:**
        1.  **Low-Activity Users:** Cost-conscious, low balance.
        2.  **Prime Transactors:** High spending, pays in full.
        3.  **Revolvers:** Moderate spending, carries debt.
        4.  **Cash Advance Users:** High risk, liquidity issues.

---

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone [Your-Repository-Link]
    ```
2.  **Dependencies:**
    * Standard ML: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`.
    * **Important:** `MidtermML-3.ipynb` requires a CUDA-enabled GPU environment (like Google Colab T4) to run the **RAPIDS** libraries (`cudf`, `cuml`).
3.  **Running:**
    * Open the notebooks in Jupyter Lab, VS Code, or Google Colab and run cells sequentially.