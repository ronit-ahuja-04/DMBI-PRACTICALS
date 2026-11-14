# DMBI CA2 – Customer Segmentation (mini project)

Streamlit application and Google Colab notebook for segmenting customers of an online retail business (UCI Online Retail dataset).

## Files
- `app.py` — Streamlit app (K-Means segmentation, PCA view, segment profiles and audience export).
- `CA2_Customer_Segmentation_Colab.ipynb` — Colab notebook of the analysis.
- `fetch_online_retail.py` — helper that downloads the dataset from the UCI repository.
- `requirements.txt`, `.streamlit/config.toml` — dependencies and app theme.

## Run the app
```bash
pip install -r requirements.txt
streamlit run app.py
```
The dataset (UCI Online Retail, id 352) is downloaded automatically on the first run and cached as `online_retail.csv`, which is not stored in this repository because of its size (about 48 MB).
