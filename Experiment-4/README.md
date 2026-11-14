# Experiment 4 – Classification using RapidMiner (Decision Tree & Naïve Bayes)

**Subject:** Data Mining and Business Intelligence (DMBI) · **Name:** Ronit Ahuja · **Roll No:** 63 · **Class:** D15A

## Aim
Classify `Churn` (Retained / Churned) on the Food Delivery Customer Churn dataset with an 80:20 split and compare Decision Tree and Naïve Bayes in Altair AI Studio.

## Files
- `DMBI_Experiment_4_Classification_RapidMiner_Ronit_Ahuja.pdf` — report (with RapidMiner screenshots)
- `Experiment_4_DecisionTree.rmp`, `Experiment_4_NaiveBayes.rmp` — RapidMiner processes
- `Food_Delivery_Customer_Churn.csv` — dataset

## Key results (100 test records)
| Model | Accuracy | Churned recall | Churned precision |
|---|---|---|---|
| Decision Tree | 68.00% | 36.84% | 63.64% |
| Naïve Bayes | 69.00% | 50.00% | 61.29% |

The Split Data operator samples randomly, so re-running the processes can change the figures slightly.
