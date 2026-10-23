"""Reproduce the final customer-level model from Notebook 4 and export the real numbers used in
the showcase graphic (ROC curves, feature importances, a sample prediction) to model_results.json.

Same data, features, customer-level split and hyperparameters as the notebook's final model.
Requires pandas + scikit-learn. Run from anywhere: python branding/source/reproduce_results.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, OrdinalEncoder, StandardScaler

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name('model_results.json')

# final model of Notebook 4 (best estimator of the customer-level grid search)
FINAL_NAME = 'Random Forest'
FINAL_ESTIMATOR = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42)

# the 11 inputs of the customer-level models (Preprocessor_function on the bigger subset)
NUM = ['Age', 'Annual_Income', 'Outstanding_Debt', 'Total_EMI_per_month', 'Delay_from_due_date',
       'Num_of_Delayed_Payment', 'Num_Bank_Accounts', 'Num_of_Loan', 'Changed_Credit_Limit']
CAT = ['Credit_Mix', 'Payment_of_Min_Amount']

feats_v2 = ['Customer_ID', 'Age', 'Annual_Income', 'Outstanding_Debt', 'Total_EMI_per_month',
            'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Credit_Utilization_Ratio',
            'Payment_Behaviour', 'Credit_Mix', 'Payment_of_Min_Amount', 'Occupation',
            'Num_Bank_Accounts', 'Num_of_Loan', 'Changed_Credit_Limit', 'Credit_Score']
data = pd.read_csv(REPO / 'data_clean/clean_dataset.csv')[feats_v2]
X = data.drop(columns=['Customer_ID', 'Credit_Score'])
y = data['Credit_Score']
groups = data['Customer_ID']

# customer-level split, as in Notebook 4: 20% of the customers are kept for testing
train_index, test_index = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups=groups))
Xtr, Xte, ytr, yte = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

model = Pipeline([('preprocessor', make_column_transformer((StandardScaler(), NUM), (OrdinalEncoder(), CAT),
                                                           remainder='drop')),
                  ('estimator', FINAL_ESTIMATOR)]).fit(Xtr, ytr)

proba = model.predict_proba(Xte)
yb = LabelBinarizer().fit(model.classes_).transform(yte)
res = {'model': FINAL_NAME, 'n_features': len(NUM) + len(CAT), 'test_customers': int(groups.iloc[test_index].nunique()),
       'train_acc': accuracy_score(ytr, model.predict(Xtr)), 'test_acc': accuracy_score(yte, model.predict(Xte)),
       'baseline': float(yte.value_counts(normalize=True).max()), 'auc': {}, 'roc': {}}
for i, c in enumerate(model.classes_):
    fpr, tpr, _ = roc_curve(yb[:, i], proba[:, i])
    res['auc'][c] = auc(fpr, tpr)
    grid = np.linspace(0, 1, 61)  # resample on an even FPR grid for plotting
    res['roc'][c] = [[float(g), float(np.interp(g, fpr, tpr))] for g in grid]

names = [n.split('__')[1] for n in model.named_steps['preprocessor'].get_feature_names_out()]
importances = sorted(zip(names, model.named_steps['estimator'].feature_importances_.tolist()), key=lambda x: -x[1])

# a confident (top 10%, not the most extreme), correctly classified 'Good' applicant for the UI mock
classes = list(model.classes_)
good = classes.index('Good')
cands = np.where((model.predict(Xte) == 'Good') & (yte.values == 'Good'))[0]
pick = cands[np.argsort(-proba[cands, good])][len(cands) // 10]
sample = {'features': {k: (v.item() if hasattr(v, 'item') else v) for k, v in Xte.iloc[pick].items()},
          'proba': dict(zip(classes, proba[pick].round(4).tolist()))}

print(f"[{FINAL_NAME}] train={res['train_acc']:.4f} test={res['test_acc']:.4f} baseline={res['baseline']:.4f} "
      f"auc={ {k: round(v, 3) for k, v in res['auc'].items()} }")
json.dump({'final': res, 'importances': importances, 'sample': sample}, open(OUT, 'w'), indent=1)
print('saved', OUT)
