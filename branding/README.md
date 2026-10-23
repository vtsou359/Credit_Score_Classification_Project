# ScoreBridge: LinkedIn project kit

Graphics and copy for presenting this project in the **Projects** section of LinkedIn.

![ScoreBridge showcase](scorebridge_showcase.png)

<img src="scorebridge_logo.png" alt="ScoreBridge logo" width="160">

| File | Use |
|---|---|
| `scorebridge_showcase.png` | Product-style presentation, 2400×1256 (LinkedIn's 1.91:1 image ratio) |
| `scorebridge_logo.png` | Logo, 1200×1200, full-bleed (safe where transparency isn't kept) |
| `scorebridge_logo.svg` | Vector logo (rounded app-icon tile) for READMEs, slides, favicons |

## LinkedIn copy

**Title**

> ScoreBridge — Credit Score Classification with Machine Learning

**Description**

> End-to-end machine-learning project that predicts a customer's credit-score tier (Poor / Standard / Good) from financial and behavioural data, built to support lending decisions and promote financial inclusion.
>
> • Cleaned 100K messy records (corrupted entries, impossible values, outliers) with customer-level imputation, and removed PII
> • Ran EDA and expert-informed feature selection, then built scikit-learn preprocessing pipelines (scaling + one-hot/ordinal encoding)
> • Benchmarked 6 model families (Logistic Regression, KNN, Decision Tree, SVM, Random Forest, Gradient Boosting) with GridSearchCV on ROC-AUC
> • Caught data leakage: with 8 monthly records per customer, a random split put the same customers in train and test (Random Forest 80% → 66% on unseen customers). Re-validated by customer with GroupShuffleSplit + GroupKFold
> • Final Random Forest: 70% accuracy on unseen customers vs a 53% baseline, ROC-AUC up to 0.87
>
> Stack: Python, pandas, scikit-learn, Matplotlib, Seaborn, Jupyter · Method: CRISP-DM

**Skills to tag (up to 5):** Machine Learning · Python (Programming Language) · Scikit-Learn · Predictive Modeling · Feature Engineering

**Also worth adding to the profile Skills section:** Data Science · Data Cleaning · Exploratory Data Analysis · Model Validation · Hyperparameter Tuning · Model Evaluation · Credit Risk Modeling · Data Visualization

## Brand

- **Primary colour: Crimson `#BE3144`**, the middle step of the class palette the EDA plots already use
  (`#F05941 → #BE3144 → #872341` in `custom_functions/utility_functions.py`).
- **Mark:** an arch drawn as a three-segment credit gauge (Poor → Standard → Good), needle on *Good*;
  the arch doubles as a bridge, the idea that ML can help bridge the financial-inclusion gap.
- **Tier colours on dark surfaces:** Poor `#B83A5B`, Standard `#E0505A`, Good `#F8906F`. These are the same hues, lifted
  so every tier clears 3:1 contrast on the dark panel. Background ink `#0D0F15`.
- **Type:** Inter / Inter Display, JetBrains Mono for labels.

## Where the numbers come from

Every chart value in the showcase is real. `source/reproduce_results.py` retrains the final model of Notebook 4, the
Random Forest tuned with customer-level cross-validation (same data, 11 features, customer-level split and
hyperparameters), and reproduces its results on the 2,500 unseen test customers: accuracy 69.9% (baseline 53.1%),
one-vs-rest ROC-AUC 0.87 / 0.85 / 0.79. The ROC curves, feature importances and the sample applicant come from that
run (`source/model_results.json`).

## Regenerate

Requires the Inter, Inter Display and JetBrains Mono fonts, and `npm i playwright`.

```bash
cd branding/source
python reproduce_results.py   # optional, needs the pinned requirements; refreshes model_results.json
python build_assets.py        # writes build/*.html and ../scorebridge_logo.svg
node render.js build/showcase.html ../scorebridge_showcase.png 1200 628 2
node render.js build/logo.html ../scorebridge_logo.png 600 600 2
node render.js crisp_dm_process.html ../../images/crisp_dm_process.png 1600 580 2
```
