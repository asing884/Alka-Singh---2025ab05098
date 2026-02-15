# ML Assignment 2 — Bank Marketing (BITS WILP)

**Student:** ALKA SINGH  
**Assignment:** Machine Learning Assignment - 2  
**Dataset:** UCI Bank Marketing (bank-additional-full.csv)  
**Task:** Classification — Predict if client subscribes to term deposit (y: yes/no)

> Dataset source & details: UCI ML Repository — Bank Marketing. (We used `bank-additional-full.csv` from `bank-additional.zip`).  
> Note: The `duration` field is excluded for realistic modeling as commonly recommended in docs based on the UCI description.  
> References:  
> - UCI dataset page: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing  
> - UCI (beta) dataset page variant with the same description: https://archive-beta.ics.uci.edu/dataset/222/bank+marketing  
> - Duration leakage note (derivative documentation): https://docs.1010data.com/Tutorials/MachineLearningExamples/BankMarketingDataSet_3.html

---

## 1) Problem Statement
Build and compare multiple ML classifiers on the Bank Marketing dataset to predict whether a client subscribes to a term deposit.

## 2) Dataset Description
- **Source:** UCI Bank Marketing (bank-additional-full.csv inside bank-additional.zip)  
- **Rows:** ~41,188; **Features:** 20 (mix of categorical & numeric); **Target:** y (0=no, 1=yes)  
- **Note:** The `duration` column is dropped to avoid information leakage.

## 3) Models Used (All on the same dataset)
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Multinomial)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### 3.1 Comparison Table (fill from `metrics_summary.md`)
Paste the Markdown table from **metrics_summary.md** here:

ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.8350	0.8009	0.3679	0.6466	0.4689	0.4011
XGBoost	0.9019	0.8035	0.6389	0.2974	0.4059	0.3916
KNN	0.8972	0.7440	0.5839	0.3039	0.3997	0.3719
Random Forest	0.8948	0.7798	0.5659	0.2823	0.3767	0.3498
Decision Tree	0.8471	0.6222	0.3245	0.3308	0.3276	0.2414
Naive Bayes	0.8977	0.6802	0.6471	0.2015	0.3073	0.3223
