# Experiment 9 – Association Rule Mining using Apriori (Python)

**Subject:** Data Mining and Business Intelligence (DMBI) · **Name:** Ronit Ahuja · **Roll No:** 63 · **Class:** D15A

## Aim
Find frequent itemsets and association rules (support, confidence, lift) from City, Subscription_Type, Favorite_Category, Device and Churn of the Food Delivery Customer Churn dataset with the Apriori algorithm (mlxtend).

## Files
- `DMBI_Experiment_9_Apriori_Association_Rule_Mining_Ronit_Ahuja.pdf` — report (code, outputs, plots, tables)
- `Experiment-9_Apriori_Association_Rule_Mining.ipynb` — executed notebook (Colab or local, with the CSV in the same folder)
- `frequent_itemsets.csv`, `association_rules.csv` — exported results
- `frequent_itemsets_graph.png`, `Experiment9_Graphs.png` — plots
- `Food_Delivery_Customer_Churn.csv` — dataset

## Key results (min support 0.10, min confidence 0.60)
- 61 frequent itemsets (16 / 39 / 6 of size 1 / 2 / 3) and 25 rules; 13 rules have a Churn value as conclusion.
- Strongest rule: Device=Android, Subscription_Type=Premium → Churn=Retained (support 0.102, confidence 0.729, lift 1.179). Lifts range from 0.978 to 1.179 (weak associations).
- Identical itemsets, supports and rules to the FP-Growth run in Experiment 8 (RapidMiner).
