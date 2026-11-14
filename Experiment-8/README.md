# Experiment 8 – Association Rule Mining using RapidMiner (FP-Growth)

**Subject:** Data Mining and Business Intelligence (DMBI) · **Name:** Ronit Ahuja · **Roll No:** 63 · **Class:** D15A

## Aim
Mine frequent itemsets and association rules from City, Subscription_Type, Favorite_Category, Device and Churn of the Food Delivery Customer Churn dataset with FP-Growth and Create Association Rules in Altair AI Studio.

## Files
- `DMBI_Experiment_8_FPGrowth_RapidMiner_Ronit_Ahuja.pdf` — report (includes AI Studio screenshots of the process and the frequent itemsets)
- `Experiment_8_FPGrowth_RapidMiner.rmp` — RapidMiner process (Retrieve → Select Attributes → Nominal to Binominal → FP-Growth → Create Association Rules → Item Sets to Data)
- `Food_Delivery_Customer_Churn.csv` — dataset

## Key results (min support 0.1, min confidence 0.6)
- 61 frequent itemsets (16 / 39 / 6 of size 1 / 2 / 3) and 25 rules; identical itemsets and supports to the Apriori run in Experiment 9.
- Strongest rule: Device = Android, Subscription_Type = Premium → Churn = Retained (support 0.102, confidence 0.729, lift 1.179). Lifts range from 0.978 to 1.179 (weak associations).

## Running the process
The `.rmp` reads the repository entry `//Local Repository/Food_Delivery_Customer_Churn (1)`. Import the CSV into the Local Repository under that name, then open and run the process.
