# Consumer Credit Risk Dashboard & Predictive Pipeline

Predicts the probability that a consumer loan will default using a Random Forest classifier, trained on ~22,900 loans and tested on 5,728 held-out loans. The scored portfolio feeds an interactive Tableau Public dashboard.

**Live dashboard:** [Tableau Public: Credit Risk Data Collection](https://public.tableau.com/app/profile/blake.konger/viz/CreditRiskDataCollection/Dashboard1?publish=yes)

**Tools:** Python (pandas, scikit-learn) · Google Colab · Tableau Public

---

## Results at a glance

Evaluated on a held-out test set of **5,728 loans** that the model never saw during training:

| Metric | Result | What it means |
|---|---|---|
| Accuracy | **93%** | vs. 77.6% for a naive model that predicts "no default" for every loan |
| Default recall | **72%** | Catches about 7 in 10 loans that actually defaulted |
| Default precision | **96%** | When it flags a loan as a default, it's right 96% of the time |
| Non-default recall | **99%** | Almost never flags a good loan as risky |

22.4% of the test loans defaulted (1,285 of 5,728).

## Why this matters in lending

A lender's core credit question is how likely a borrower is to default. That is the **probability of default (PD)**. PD is one input to expected loss (PD × loss given default × exposure at default), which drives pricing, approval cutoffs, and loss reserves. This project produces a PD-style score for every loan and makes the portfolio's risk easy to explore visually.

## Data

[Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (Kaggle): 32,581 consumer loans, 12 columns.

| Column | Description |
|---|---|
| `person_age` | Borrower age |
| `person_income` | Annual income ($) |
| `person_home_ownership` | RENT, OWN, MORTGAGE, or OTHER |
| `person_emp_length` | Years of employment |
| `loan_intent` | Loan purpose: debt consolidation, education, home improvement, medical, personal, or venture |
| `loan_grade` | Lender-assigned grade, from A (safest) to G |
| `loan_amnt` | Loan amount ($) |
| `loan_int_rate` | Interest rate (%) |
| `loan_percent_income` | Loan amount ÷ annual income |
| `cb_person_default_on_file` | Prior default on the borrower's credit bureau file (Y/N) |
| `cb_person_cred_hist_length` | Length of credit history (years) |
| `loan_status` | **Target:** 1 = defaulted, 0 = did not default |

## How the pipeline works

1. **Load** `credit_risk_dataset.csv` with pandas (32,581 rows).
2. **Clean:** drop rows with missing values. The gaps were in `loan_int_rate` (3,116 rows) and `person_emp_length` (895 rows). About 28,600 loans remain.
3. **Encode:** one-hot encode the four text columns with `pd.get_dummies(drop_first=True)`, giving 22 numeric model features.
4. **Split:** 80% training and 20% test (`random_state=42` makes the results reproducible).
5. **Train:** `RandomForestClassifier(n_estimators=100)` builds 100 decision trees, each on a random sample of the training loans, and averages their predictions.
6. **Evaluate:** run `classification_report` on the test set (results above).
7. **Score:** add `Predicted_Risk_Score`, the model's predicted probability of default (0 to 1), to every loan. Loans removed during cleaning are left blank.
8. **Export:** save `credit_risk_tableau_ready.csv`, which feeds the Tableau dashboard.

Full test-set output:

```
              precision    recall  f1-score   support

           0       0.92      0.99      0.96      4443
           1       0.96      0.72      0.82      1285

    accuracy                           0.93      5728
   macro avg       0.94      0.86      0.89      5728
weighted avg       0.93      0.93      0.93      5728
```

## Dashboard

The Tableau Public dashboard is built on the scored export and has three views:

- **Portfolio summary:** 32,581 loans, about $312M in total loan amount, an average predicted default risk of 21.8%, and the approval rate (the share of scored loans with predicted risk below a 50% cutoff).
- **Predicted risk by loan purpose:** debt consolidation, medical, and home improvement loans carry the highest average predicted risk (about 26–27%); venture loans carry the lowest (about 15%).
- **Income vs. loan amount:** each loan plotted by borrower income and loan size, colored by predicted risk. High-risk loans concentrate among lower-income borrowers.

## Limitations and next steps

- **Tune the cutoff.** A loan is flagged as a default when its predicted probability is above 0.5, which misses about 28% of defaulters. Approving a borrower who defaults usually costs more than turning down a good one, so the next step is to choose the cutoff based on those costs and to report ROC-AUC, Gini, and KS.
- **Score the dashboard out of sample.** The export scores every loan, including the 80% the model trained on, so those scores look more confident than they would on new loans. Out-of-fold predictions (`cross_val_predict`) would fix this.
- **Check calibration.** In credit risk, a predicted 10% PD should mean that about 10% of those loans actually default. A calibration curve would test this.
- **Handle missing data instead of dropping it.** Dropping rows removed about 12% of the loans. Median imputation plus a "was missing" flag would keep them.
- **Clean outliers.** A few values are impossible. For example, the first row lists 123 years of employment for a 22-year-old.
- **Test without lender-assigned fields.** `loan_grade` and `loan_int_rate` already reflect the lender's own risk assessment, so the model partly relies on an existing score.
- **Explain and benchmark.** Add feature importances, and compare against logistic regression, the more interpretable standard for bank credit models.

## How to run

1. Download `credit_risk_dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset).
2. Open `Collab-Data.ipynb` in Google Colab or Jupyter, upload the CSV to the session, and run all cells.
3. Requirements: Python 3, `pandas`, and `scikit-learn`.

`Collab-Code.py` is the same code exported from Colab. It uses Colab's `display()` function, so run it in a notebook, or replace `display` with `print`.

## Repository contents

| File | Description |
|---|---|
| `Collab-Data.ipynb` | Notebook with the code and saved outputs |
| `Collab-Code.py` | The same pipeline as a Python script |

---

**Blake Konger** · Purdue University · B.S. Mathematics, minors in Computer Science & Finance · [LinkedIn](https://www.linkedin.com/in/blake-konger-5b65a4397)
