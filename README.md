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


|index|ML Model Name|Accuracy|AUC|Precision|Recall|F1|MCC|
|---|---|---|---|---|---|---|---|
|0|Logistic Regression|0\.8350327749453751|0\.80094145065805|0\.3678724708767627|0\.646551724137931|0\.46893317702227433|0\.40108374113193157|
|5|XGBoost|0\.9019179412478757|0\.8034678080333978|0\.6388888888888888|0\.2974137931034483|0\.40588235294117647|0\.3915621658427986|
|2|KNN|0\.8971837824714737|0\.7440191754328035|0\.5838509316770186|0\.30387931034482757|0\.3997165131112686|0\.37194533230817917|
|4|Random Forest|0\.8947560087399854|0\.7797993124675692|0\.5658747300215983|0\.2823275862068966|0\.3767074047447879|0\.34981969052018386|
|1|Decision Tree|0\.8470502549162418|0\.6222374876173404|0\.32452431289640593|0\.3308189655172414|0\.327641408751334|0\.24137501776047923|
|3|Naive Bayes|0\.8976693372177713|0\.6801583358884854|0\.6470588235294118|0\.20150862068965517|0\.3073130649137223|0\.3222971458261065|

### 3.2 Observations — Model-by-Model (Fill after reviewing metrics)

| **ML Model Name**            | **Observation about model performance** |
|-----------------------------|-----------------------------------------|
| Logistic Regression         | Strong linear baseline; good AUC and stable precision/recall with class weighting; may underfit complex non-linear patterns. |
| Decision Tree               | Interpretable but higher variance; without pruning can overfit; generally lower AUC/F1 than ensembles. |
| kNN                         | Sensitive to scaling and high-dimensional one-hot space; moderate metrics; inference is costlier. |
| Naive Bayes (Multinomial)   | Very fast and OHE-friendly; reasonable recall but typically lower precision/AUC vs linear/ensemble models. |
| Random Forest (Ensemble)    | Robust, captures interactions well; usually strong F1/AUC out-of-the-box; useful feature importance. |
| XGBoost (Ensemble)          | Often best F1/AUC; models subtle interactions; benefits from tuning; strong precision-recall balance. |
