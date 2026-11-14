# Experiment 7 – Clustering using RapidMiner (k-Means, Agglomerative, DBSCAN)

**Subject:** Data Mining and Business Intelligence (DMBI) · **Name:** Ronit Ahuja · **Roll No:** 63 · **Class:** D15A

## Aim
Segment the Food Delivery Customer Churn dataset in Altair AI Studio with k-Means, Agglomerative clustering and DBSCAN (Select Attributes → Replace Missing Values → Z-score Normalize → clustering) and compare the results.

## Files
- `DMBI_Experiment_7_Clustering_RapidMiner_Ronit_Ahuja.pdf` — report
- `Experiment_7_Clustering_RapidMiner.rmp` — RapidMiner process with all three algorithms
- `Experiment_7_KMeans_RapidMiner.rmp` — RapidMiner process with k-Means only
- `Food_Delivery_Customer_Churn.csv` — dataset

## Key results (RapidMiner, 4 normalized numeric attributes)
| Algorithm | Parameters | Clusters | Noise | Silhouette |
|---|---|---|---|---|
| k-Means | k = 3 | 164 / 165 / 171 | – | 0.1856 |
| Agglomerative | Average Link, 3 clusters | 165 / 115 / 220 | – | 0.1452 |
| DBSCAN | eps 0.8, minPts 5 | 4 clusters (439, 5, 5, 7) | 44 | −0.0305 |

k-Means gives the most useful segmentation; DBSCAN suits outlier detection more than segmentation on this evenly spread data.

## Running the processes
The `.rmp` files read the dataset from the AI Studio repository entry `//Local Repository/Food_Delivery_Customer_Churn (1)`. Import `Food_Delivery_Customer_Churn.csv` into the Local Repository under that name (or change the Retrieve operator) and run.

## DBSCAN epsilon check (RapidMiner, minPts = 5)
| Epsilon | Clusters | Noise points |
|---|---|---|
| 0.5 | 5 | 469 |
| 0.6 | 17 | 352 |
| 0.7 | 8 | 163 |
| 0.8 | 4 | 44 |
| 0.9 | 2 | 4 |
| 1.0 | 1 | 1 |
