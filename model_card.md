# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This implementation uses a scikit-learn RandomForestClassifier for binary income classification. The explicitly configured parameters are n_estimators=200, min_samples_leaf=2, random_state=42, max_depth=None, and n_jobs=-1. Other parameters use the installed scikit-learn version's defaults.

The preprocessing function one-hot encodes eight categorical features: workclass, education, marital-status, occupation, relationship, race, sex, and native-country. It uses handle_unknown="ignore", so unseen categorical values are encoded without raising an unknown-category error. The remaining feature columns are passed through without scaling. A LabelBinarizer encodes the target.

The implementation saves the classifier, categorical encoder, and label binarizer as model.joblib, encoder.joblib, and lb.joblib.


## Intended Use

This model is intended for an educational income-classification project and offline evaluation on the supplied Census data. No production readiness or fairness certification is claimed.

## Training Data

The training script reads data/census.csv. It replaces missing categorical values and blank categorical strings with "?", strips categorical whitespace, and converts salary labels to stripped strings. Existing "?" categories are retained. The supplied loader does not remove duplicate rows or explicitly reject missing target labels before converting them to strings.

The script assigns approximately 80 percent of rows to training using train_test_split. Encoders are fitted only on the training partition.

## Evaluation Data

The script reserves 20 percent of the input for held-out evaluation.

Evaluation reuses the fitted training encoders. The slice function is intended to evaluate each category observed in the test partition for every categorical feature. Categories absent from the test partition are omitted. The intended output is slice_output.txt, containing each slice's feature, value, sample count, precision, recall, and fbeta.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
The evaluation metrics are
- **Precision: Measures the proportion of positive predictions that are actually correct (True Positives / (True Positives + False Positives))
- **Recall: Measures the proportion of actual positive cases that are correctly identified (True Positives / (True Positives + False Negatives))
- **F-beta Score (F1): Harmonic mean of precision and recall, providing a balanced measure of model performance

## Ethical Considerations

The input features include race and sex, and other variables can act as proxies for protected characteristics. This creates potential fairness risks. 
Slice evaluation can reveal performance variation but does not by itself establish fairness, especially when slices contain few examples or no positive examples. 
No inference about the merit, suitability, or worth of individuals should be drawn from this model.
Any real-world use would require appropriate governance, privacy review, and independent validation.

## Caveats and Recommendations

- The model is trained on data from 1994, making it potentially outdated for current predictions
- The binary classification of income (<=50K vs >50K) is a simplification that may not capture economic complexity
- This model should only be used for educational and demonstration purposes