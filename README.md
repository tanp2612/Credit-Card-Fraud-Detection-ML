# 💳 Credit Card Fraud Detection using Machine Learning

> **An end-to-end Machine Learning project for detecting fraudulent credit card transactions in a highly imbalanced dataset.**

---

## 📌 Overview

Credit card fraud is a major challenge for financial institutions. With millions of transactions processed every day, manually identifying suspicious activity is inefficient and error-prone.

This project focuses on building **Machine Learning models capable of distinguishing fraudulent transactions from legitimate ones**, with particular emphasis on handling the **extreme class imbalance** present in real-world fraud detection datasets.

The project uses a publicly available dataset containing **284,807 credit card transactions**, of which only **492 are fraudulent**.

### 🎯 Key Challenge

Only **0.172%** of transactions belong to the fraud class.

This means that simply predicting every transaction as legitimate would result in extremely high accuracy while completely failing at the actual task of detecting fraud.

Therefore, the project focuses on:

* 🔹 Handling severe class imbalance
* 🔹 Exploring transaction patterns
* 🔹 Applying appropriate sampling techniques
* 🔹 Training and tuning multiple ML models
* 🔹 Evaluating models using fraud-focused metrics

---

## 🏦 Business Context

For financial institutions, retaining customers and maintaining trust are critical priorities.

Fraud can result in:

* 💰 Direct financial losses
* 🔄 Costly chargebacks
* ⏱️ Increased manual investigation
* 😟 Customer dissatisfaction
* 🏦 Damage to institutional reputation

As digital payment systems continue to grow, fraudulent activity has also become increasingly sophisticated.

Machine Learning can help financial institutions **automatically identify suspicious transactions**, allowing potentially fraudulent activity to be flagged for further investigation while reducing unnecessary rejection of legitimate transactions.

---

## 🎯 Problem Statement

The primary objective of this project is to develop Machine Learning models that can accurately identify fraudulent credit card transactions.

The project specifically addresses the challenge of **highly imbalanced classification**, where fraudulent transactions represent only a very small fraction of the overall dataset.

The goal is therefore not simply to maximize accuracy, but to build models that can effectively identify the **minority fraud class** while maintaining reasonable performance on legitimate transactions.

---

## 📊 Dataset

The dataset contains credit card transactions made by European cardholders over **two days in September 2013**.

### Dataset Statistics

| Property                |       Value |
| ----------------------- | ----------: |
| Total Transactions      | **284,807** |
| Fraudulent Transactions |     **492** |
| Legitimate Transactions | **284,315** |
| Fraud Percentage        |  **0.172%** |
| Features                |      **30** |

### 🔐 Data Privacy

To protect customer confidentiality, the original features were transformed using **Principal Component Analysis (PCA)**.

As a result:

* `V1` – `V28` → PCA-transformed features
* `Time` → Seconds elapsed between transactions
* `Amount` → Transaction amount
* `Class` → Target variable

### Target Variable

```text
Class = 0 → Legitimate Transaction
Class = 1 → Fraudulent Transaction
```

---

## 🚨 Types of Credit Card Fraud

Credit card fraud refers to the unauthorized use of card or payment information for financial gain.

Common forms include:

* 💳 **Card skimming** — stealing card information through compromised payment devices
* 🪪 **Counterfeit cards** — creating unauthorized copies of legitimate cards
* 📦 **Lost or stolen cards** — using a physical card without the owner's permission
* 📞 **Deceptive schemes** — obtaining card information through fraudulent communication
* 🔐 **Compromised card information** — using stolen credentials for unauthorized transactions

---

# 🔬 Project Pipeline

```text
             ┌──────────────────────┐
             │   Dataset Loading    │
             └──────────┬───────────┘
                        ↓
             ┌──────────────────────┐
             │   Data Exploration   │
             │        & EDA         │
             └──────────┬───────────┘
                        ↓
             ┌──────────────────────┐
             │ Train / Test Split   │
             │ & Cross Validation   │
             └──────────┬───────────┘
                        ↓
             ┌──────────────────────┐
             │ Imbalance Handling   │
             │  & Sampling          │
             └──────────┬───────────┘
                        ↓
             ┌──────────────────────┐
             │ Model Development    │
             │   & Hyperparameter   │
             │       Tuning         │
             └──────────┬───────────┘
                        ↓
             ┌──────────────────────┐
             │ Model Evaluation     │
             │ & Comparison         │
             └──────────────────────┘
```

---

## 1️⃣ Data Familiarization

The dataset is loaded and examined to understand:

* Dataset dimensions
* Feature distributions
* Missing values
* Class distribution
* Transaction amount patterns
* Relationships between features and fraud

Understanding the data is especially important because the fraud class represents only a tiny portion of the dataset.

---

## 2️⃣ Exploratory Data Analysis

Exploratory Data Analysis is performed to identify patterns and characteristics of fraudulent transactions.

### Analysis includes:

* 📈 Univariate analysis
* 📊 Bivariate analysis
* 🔍 Class distribution
* 💰 Transaction amount analysis
* ⏱️ Transaction time analysis
* 🔗 Feature relationships

Since most of the variables are already PCA-transformed, traditional feature scaling requirements may differ from those of raw-feature datasets.

---

## 3️⃣ Data Splitting & Validation

The dataset is divided into **training and testing sets** to evaluate how well the models generalize to unseen transactions.

Because fraud cases are extremely rare, **stratified validation techniques** are important to ensure that fraudulent transactions are adequately represented across training and validation subsets.

### Key considerations:

* Train/Test split
* Stratified sampling
* Cross-validation
* Avoiding data leakage

---

## 4️⃣ Handling Class Imbalance

One of the biggest challenges in this project is the severe imbalance between legitimate and fraudulent transactions.

### Class distribution:

```text
Legitimate  █████████████████████████████████████████████  99.828%
Fraud       ▏                                             0.172%
```

To improve the models' ability to learn fraudulent patterns, imbalance-handling techniques such as **minority-class oversampling** can be explored.

The objective is to give the model sufficient exposure to fraudulent examples without compromising evaluation reliability.

---

## 5️⃣ Model Development & Tuning

Multiple Machine Learning algorithms can be experimented with and compared.

The general workflow involves:

```text
Model Selection
      ↓
Baseline Training
      ↓
Imbalance Handling
      ↓
Hyperparameter Tuning
      ↓
Cross Validation
      ↓
Final Model
```

Hyperparameters are optimized to improve the model's ability to detect fraudulent transactions while maintaining acceptable performance on legitimate transactions.

---

# 📏 Model Evaluation

Accuracy alone is **not an appropriate metric** for this problem.

For example, a model that predicts every transaction as legitimate could achieve approximately **99.8% accuracy**, while detecting **zero fraudulent transactions**.

Therefore, fraud-focused evaluation metrics are more meaningful.

### Key Metrics

| Metric               | Why it matters                                                             |
| -------------------- | -------------------------------------------------------------------------- |
| **Precision**        | Measures how many transactions flagged as fraud are actually fraudulent    |
| **Recall**           | Measures how many actual fraudulent transactions are successfully detected |
| **F1-Score**         | Balances Precision and Recall                                              |
| **ROC-AUC**          | Measures overall classification performance across thresholds              |
| **Confusion Matrix** | Shows True Positives, True Negatives, False Positives and False Negatives  |

### 🚨 Why Recall Matters

In fraud detection, a **False Negative** means that an actual fraudulent transaction was classified as legitimate.

Missing fraudulent transactions can be significantly more costly than investigating some legitimate transactions that were incorrectly flagged.

Therefore, the project places particular emphasis on the model's ability to identify the **fraudulent class**.

---

# 🛠️ Technologies Used

| Technology                  | Purpose                       |
| --------------------------- | ----------------------------- |
| 🐍 **Python**               | Programming language          |
| 🧮 **NumPy**                | Numerical computation         |
| 🐼 **Pandas**               | Data manipulation             |
| 📊 **Matplotlib / Seaborn** | Data visualization            |
| 🤖 **Scikit-learn**         | Machine Learning              |
| 📓 **Jupyter Notebook**     | Development & experimentation |
| 📁 **Kaggle Dataset**       | Data source                   |

---

# 📂 Project Structure

```text
Credit-Card-Fraud-Detection-ML/
│
├── 📓 Credit_Card_Fraud_Detection.ipynb
├── 📄 README.md
├── 📊 creditcard.csv
└── 📁 images/
    ├── class_distribution.png
    ├── correlation_matrix.png
    └── model_results.png
```

> *Update the filenames above to match the actual files in your repository.*

---

# 💡 Key Takeaways

### 🔹 Extreme class imbalance matters

Fraudulent transactions represent only a tiny fraction of all transactions, making conventional classification approaches insufficient.

### 🔹 Accuracy can be misleading

A model can achieve very high accuracy while completely failing to detect fraud.

### 🔹 Sampling is important

Techniques for handling class imbalance can help models learn meaningful patterns from the minority class.

### 🔹 Evaluation should reflect business costs

Metrics such as **Precision, Recall and F1-Score** provide a more useful picture of fraud detection performance than accuracy alone.

---

# 🚀 Future Improvements

Potential extensions to this project include:

* [ ] Experiment with additional sampling strategies
* [ ] Compare more classification algorithms
* [ ] Perform systematic hyperparameter optimization
* [ ] Tune the classification threshold
* [ ] Analyze Precision-Recall curves
* [ ] Build an interactive fraud detection dashboard
* [ ] Deploy the final model as an API
* [ ] Monitor model performance over time

---

# 📚 Dataset Source

The dataset was originally made available through Kaggle and was created from transactions collected by European cardholders in September 2013.

The dataset is commonly associated with the collaborative work of **Worldline and the Machine Learning Group (ULB)**.

---

## ⭐ Project Goal

> **Build a Machine Learning system that doesn't just achieve high accuracy, but actually learns to identify fraudulent transactions hidden within millions of legitimate-looking transactions.**

---

### 👩‍💻 Author

**Tanvi Patil**

📌 Machine Learning | Data Science | Artificial Intelligence

---

⭐ **If you found this project useful, consider giving the repository a star!**
