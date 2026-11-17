# Loan Approval Prediction

A small machine learning project that predicts whether a loan application will be approved. I made it for Assignment 1 of Introduction to Data Science at Namal University (April 2026). It compares K-Nearest Neighbors and Gaussian Naive Bayes on 614 loan applications.

## The data

The file `loan-prediction-dataset.csv` has 614 applications, with 11 input columns and one target, `Loan_Status` (Y = approved, N = rejected). The inputs are gender, marital status, dependents, education, self-employment, applicant income, co-applicant income, loan amount, loan term, credit history and property area. I dropped `Loan_ID` because it is only an identifier.


## What I did

1. **Missing values.** Filled the numeric columns (loan amount, loan term, credit history) with the median. Filled the category columns (gender, married, dependents, self-employed) with the most common value. There were no duplicate rows.
2. **Encoding.** Turned the text columns into numbers with `LabelEncoder`.
3. **Split.** 80% training (491 rows) and 20% testing (123 rows), with `random_state=42`.
4. **Scaling.** Standardised the features with `StandardScaler`, fitted on the training set only and then applied to the test set, so no test information leaks into training.
5. **Models.** KNN with K = 3, 5 and 7, and Gaussian Naive Bayes.
6. **Evaluation.** Accuracy, precision, recall and F1-score on the test set.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| KNN (K=3) | 0.748 | 0.748 | 0.748 | 0.726 |
| KNN (K=5) | 0.756 | 0.768 | 0.756 | 0.729 |
| KNN (K=7) | 0.780 | 0.807 | 0.780 | 0.754 |
| Gaussian Naive Bayes | 0.780 | 0.807 | 0.780 | 0.754 |

KNN with K=7 and Gaussian Naive Bayes tied for the best result, and accuracy went up as K went up. For comparison, 80 of the 123 test applications were approved, so always predicting "approved" would give about 65% accuracy. All four models beat that, but not by a huge margin.

Precision, recall and F1 are weighted averages over both classes.

## Limits of this work

- The test set is only 123 rows and I used one split. One test row is about 0.8% of accuracy, so the gaps between these models are only a few predictions. A different random seed could change the ranking.
- I used label encoding for all category columns, which is a simple choice. One-hot encoding might work better, especially for Naive Bayes.

## How to run

Keep `loan-prediction-dataset.csv` in the same folder as the notebook, open `Haseeb_assignment1.ipynb` in Jupyter, and run the cells from top to bottom. You need Python with pandas, numpy and scikit-learn.