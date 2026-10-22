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
> • Benchmarked 6 model families (Logistic Regression, KNN, Decision Tree, SVM, Random Forest, Gradient Boosting), tuned with GridSearchCV and k-fold cross-validation on ROC-AUC
> • Final Gradient Boosting model: 74% test accuracy vs a 53% baseline, with one-vs-rest ROC-AUC up to 0.91
>
> Stack: Python, pandas, scikit-learn, Matplotlib, Seaborn, Jupyter · Method: CRISP-DM

**Skills to tag (up to 5):** Machine Learning · Python (Programming Language) · Scikit-Learn · Predictive Modeling · Feature Engineering

**Also worth adding to the profile Skills section:** Data Science · Data Cleaning · Exploratory Data Analysis · pandas · Classification · Gradient Boosting · Hyperparameter Tuning · Model Evaluation · Credit Risk Modeling · Data Visualization

## Brand

- **Primary colour: Crimson `#BE3144`**, the middle step of the class palette the EDA plots already use
  (`#F05941 → #BE3144 → #872341` in `custom_functions/utility_functions.py`).
- **Mark:** an arch drawn as a three-segment credit gauge (Poor → Standard → Good), needle on *Good*;
  the arch doubles as a bridge, the idea that ML can help bridge the financial-inclusion gap.
- **Tier colours on dark surfaces:** Poor `#B83A5B`, Standard `#E0505A`, Good `#F8906F`. These are the same hues, lifted
  so every tier clears 3:1 contrast on the dark panel. Background ink `#0D0F15`.
- **Type:** Inter / Inter Display, JetBrains Mono for labels.

## Where the numbers come from

Every chart value in the showcase is real. `source/reproduce_results.py` retrains the final Gradient Boosting
pipeline from Notebook 4 (same features, split and hyperparameters) and reproduces the README results:
test accuracy 0.74, one-vs-rest AUC 0.91 / 0.89 / 0.82. The ROC curves, feature importances and the sample
applicant come from that run (`source/model_results.json`). The script also reports a customer-grouped split
(no customer in both train and test) for comparison.

## Regenerate

Requires the Inter, Inter Display and JetBrains Mono fonts, and `npm i playwright`.

```bash
cd branding/source
python reproduce_results.py   # optional, needs pandas + scikit-learn; refreshes model_results.json
python build_assets.py        # writes build/*.html and ../scorebridge_logo.svg
node render.js build/showcase.html ../scorebridge_showcase.png 1200 628 2
node render.js build/logo.html ../scorebridge_logo.png 600 600 2
```
