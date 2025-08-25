
# Titanic App Pro — Y. Kumar

Visual EDA, training metrics (accuracy, precision, recall, F1, ROC AUC), diagnostics (confusion matrix, ROC curve, feature importances), and downloadable predictions — all in one app.

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Uses RandomForestClassifier with engineered features (FamilySize, IsAlone, Title).
- Works with bundled `data/train.csv` and `data/test.csv`, or your own datasets.
