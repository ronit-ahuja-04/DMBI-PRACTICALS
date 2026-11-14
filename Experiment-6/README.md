# Experiment 6 – Customer Segmentation using K-Means Clustering

**Subject:** Data Mining and Business Intelligence (DMBI) · **Name:** Ronit Ahuja · **Roll No:** 63 · **Class:** D15A

## Aim
K-Means on Age, Orders_Per_Month, Average_Order_Value and Tenure_Months of the Food Delivery Customer Churn dataset, with the Elbow Method and silhouette / Davies-Bouldin evaluation.

## Files
- `DMBI_Experiment_6_Ronit_Ahuja.pdf` — report
- `Experiment-6_KMeans_Clustering.ipynb` — executed notebook (Colab or local, with the CSV in the same folder)
- `Food_Delivery_Customer_Churn.csv` — dataset

## Key results
- K = 3; clusters of 184 / 164 / 152 customers; silhouette 0.1869; inertia 1353.65; Davies-Bouldin 1.6517.
- The inertia curve has no sharp elbow and silhouette scores are similar for all K (0.19–0.24); K = 3 was chosen as an interpretable trade-off (see the K-selection table in the notebook and report).
- Low silhouette means the segments overlap and are best read as practical personas.
