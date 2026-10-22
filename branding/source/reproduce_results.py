"""Reproduce the final Gradient Boosting model from Notebook 4 and export the real numbers used in
the showcase graphic (ROC curves, feature importances, a sample prediction) to model_results.json.

Also evaluates the same model with a customer-grouped split (no customer in both train and test).

Requires pandas + scikit-learn. Run from anywhere: python branding/source/reproduce_results.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, OrdinalEncoder, StandardScaler

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name('model_results.json')

NUM = ['Age', 'Annual_Income', 'Outstanding_Debt', 'Total_EMI_per_month',
       'Delay_from_due_date', 'Num_of_Delayed_Payment']
CAT = ['Credit_Mix', 'Payment_of_Min_Amount']

# Notebook 4 fits the ordinal preprocessor on the small subset (these 8 columns, remainder='drop')
# and reuses it for the 100k models, so the final models see exactly these 8 inputs.
small = pd.read_csv(REPO / 'data_clean/for_models_dataset.csv')
pre = make_column_transformer((StandardScaler(), NUM), (OrdinalEncoder(), CAT), remainder='drop').fit(small)

feats_v2 = ['Customer_ID', 'Age', 'Annual_Income', 'Outstanding_Debt', 'Total_EMI_per_month',
            'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Credit_Utilization_Ratio',
            'Payment_Behaviour', 'Credit_Mix', 'Payment_of_Min_Amount', 'Occupation',
            'Num_Bank_Accounts', 'Num_of_Loan', 'Changed_Credit_Limit', 'Credit_Score']
data = pd.read_csv(REPO / 'data_clean/clean_dataset.csv')
X = data[feats_v2].drop(columns=['Customer_ID', 'Credit_Score'])
y = data['Credit_Score']
groups = data['Customer_ID']


def make_model():
    # best grid_gbc_v2 parameters from Notebook 4
    return Pipeline([('preprocessor', pre),
                     ('estimator', GradientBoostingClassifier(criterion='squared_error', learning_rate=1,
                                                              n_estimators=150, random_state=42))])


def evaluate(model, Xtr, ytr, Xte, yte):
    proba = model.predict_proba(Xte)
    yb = LabelBinarizer().fit(model.classes_).transform(yte)
    res = {'train_acc': accuracy_score(ytr, model.predict(Xtr)),
           'test_acc': accuracy_score(yte, model.predict(Xte)),
           'baseline': float(yte.value_counts(normalize=True).max()), 'auc': {}, 'roc': {}}
    for i, c in enumerate(model.classes_):
        fpr, tpr, _ = roc_curve(yb[:, i], proba[:, i])
        res['auc'][c] = auc(fpr, tpr)
        grid = np.linspace(0, 1, 61)  # resample on an even FPR grid for plotting
        res['roc'][c] = [[float(g), float(np.interp(g, fpr, tpr))] for g in grid]
    return res, proba


# 1) Row-level stratified split, exactly as in Notebook 4
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)
gb = make_model().fit(Xtr, ytr)
row_res, proba = evaluate(gb, Xtr, ytr, Xte, yte)

names = [n.split('__')[1] for n in gb.named_steps['preprocessor'].get_feature_names_out()]
importances = sorted(zip(names, gb.named_steps['estimator'].feature_importances_.tolist()), key=lambda x: -x[1])

# A confident (not extreme), correctly classified 'Good' applicant for the UI mock
classes = list(gb.classes_)
good = classes.index('Good')
cands = np.where((gb.predict(Xte) == 'Good') & (yte.values == 'Good'))[0]
pick = cands[np.argsort(-proba[cands, good])][len(cands) // 6]
sample = {'features': {k: (v.item() if hasattr(v, 'item') else v) for k, v in Xte.iloc[pick].items()},
          'proba': dict(zip(classes, proba[pick].round(4).tolist()))}

# 2) Customer-grouped split
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups))
grp_res, _ = evaluate(make_model().fit(X.iloc[tr], y.iloc[tr]), X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te])
grp_res.pop('roc')

for name, r in (('row split', row_res), ('group split', grp_res)):
    print(f"[{name}] train={r['train_acc']:.4f} test={r['test_acc']:.4f} baseline={r['baseline']:.4f} "
          f"auc={ {k: round(v, 3) for k, v in r['auc'].items()} }")

json.dump({'row_split': row_res, 'group_split': grp_res, 'importances': importances, 'sample': sample},
          open(OUT, 'w'), indent=1)
print('saved', OUT)
