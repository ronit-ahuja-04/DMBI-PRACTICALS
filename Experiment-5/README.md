# Experiment 5 – Classification using Python (Decision Tree & Naïve Bayes) – Inbuilt vs User-Defined

**Subject:** Data Mining and Business Intelligence (DMBI) · **Name:** Ronit Ahuja · **Roll No:** 63 · **Class:** D15A

## Aim
Classify `Churn` on the Food Delivery Customer Churn dataset with Decision Tree (Gini / Entropy) and Gaussian Naïve Bayes, using library calls and user-defined functions, and compare accuracy, precision, recall and confusion matrices.

## Files
- `DMBI Experiment 5 classification using python.pdf` — report
- `Experiment-5_Classification_Decision_Tree_Naive_Bayes.ipynb` — executed notebook (runs in Colab or locally with the CSV in the same folder)
- `Food_Delivery_Customer_Churn.csv` — dataset

## Key results (100 test records, 80:20 stratified split)
| Classifier | Accuracy | Churned recall |
|---|---|---|
| Inbuilt Decision Tree (Gini) | 72% | 0.50 |
| User-defined Decision Tree (Entropy) | 74% | 0.47 |
| Inbuilt Naïve Bayes | 74% | 0.58 |
| User-defined Naïve Bayes | 74% | 0.58 |
