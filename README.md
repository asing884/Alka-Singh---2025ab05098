# Assignment Name
Machine Learning Assignment - 2 

## Problem Statement
Build and compare multiple ML classifiers on the UCI Bank Marketing dataset to predict whether a client will subscribe to a term deposit.

## Dataset Description
- Source: UCI Bank Marketing (file: bank-additional-full.csv inside bank-additional.zip)
- Rows: ~41,188
- Features: 20 (mix of categorical & numeric)
- Target: y (0 = no, 1 = yes)
- Note: The column `duration` is excluded during modeling to avoid leakage.

## Models Used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Multinomial)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

## Model Comparison Table

|ML Model Name|Accuracy|AUC|Precision|Recall|F1|MCC|
|---|---|---|---|---|---|---|
|Random Forest|0\.8897790725904345|0\.7985153338600879|0\.510593220338983|0\.5193965517241379|0\.5149572649572649|0\.45280869352549596|
|Logistic Regression|0\.8350327749453751|0\.80094145065805|0\.3678724708767627|0\.646551724137931|0\.46893317702227433|0\.40108374113193157|
|XGBoost|0\.9019179412478757|0\.8034678080333978|0\.6388888888888888|0\.2974137931034483|0\.40588235294117647|0\.3915621658427986|
|KNN|0\.8971837824714737|0\.7440191754328035|0\.5838509316770186|0\.30387931034482757|0\.3997165131112686|0\.37194533230817917|
|Decision Tree|0\.8470502549162418|0\.6222374876173404|0\.32452431289640593|0\.3308189655172414|0\.327641408751334|0\.24137501776047923|
|Naive Bayes|0\.8976693372177713|0\.6801583358884854|0\.6470588235294118|0\.20150862068965517|0\.3073130649137223|0\.3222971458261065|
## Model Observation Table

| ML Model Name             | Observation about model performance |
|---------------------------|-------------------------------------|
| Logistic Regression       | Good linear baseline; stable AUC and precision/recall with class weighting; may underfit complex patterns. |
| Decision Tree             | Interpretable but higher variance; without pruning can overfit; generally lower AUC/F1 than ensembles. |
| KNN                       | Sensitive to scaling and high‑dimensional one‑hot space; moderate metrics; inference is costlier. |
| Naive Bayes (Multinomial) | Very fast and OHE‑friendly; reasonable recall but typically lower precision/AUC vs linear/ensemble models. |
| Random Forest (Ensemble)  | Robust, captures interactions; usually strong F1/AUC out‑of‑the‑box; useful feature importance. |
| XGBoost (Ensemble)        | Often best F1/AUC; models subtle interactions; benefits from tuning; strong precision–recall balance. |
