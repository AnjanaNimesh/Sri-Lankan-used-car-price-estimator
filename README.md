# Sri Lankan used-car price estimator : ensemble learning

Hands-On Ensemble Learning Project: **Bagging vs Boosting vs Stacking vs Hybrid** on a local Sri Lankan dataset.

* **Live dashboard:** _<paste your Streamlit Community Cloud URL here>_
* **Notebook:** [`SL_Vehicle_Price_Ensemble_Learning.ipynb`](SL_Vehicle_Price_Ensemble_Learning.ipynb)

## Dataset
10,388 car adverts scraped from **ikman.lk** (Sri Lanka's largest classifieds site): brand, model, edition/grade,
year of manufacture, transmission, fuel, body type, engine capacity, **real odometer mileage**, asking price (Rs) and
a "negotiable" flag. Prices are modelled in **LKR lakhs** (1 lakh = Rs. 100,000). Source: _<add scrape date / link>_.

**Cleaning** (10,388 → 8,573 adverts)
* 1,814 exact duplicate adverts removed (17 %).
* 1 price typo removed (a Mazda 3 at Rs 21.5 billion).
* 19 mileage placeholders fixed: about 1,000,000 km, or 0 km on old cars. Each is replaced by the median mileage for that year.
* 19 engine-size typos fixed (e.g. 1 cc, or 10,000 cc for a Micro Panda). Each is replaced by the median for that model. Electric cars are set to 0 cc.
* Missing body type is labelled "Unknown". Text was standardised.

## Models
| Paradigm | Model |
|---|---|
| Bagging | Random Forest |
| Boosting | XGBoost |
| Stacking | RF + XGBoost + Ridge → Ridge meta-learner (5-fold OOF) |
| Hybrid | RF + ANN weighted blend (weight chosen on OOF predictions); RF → ANN residual correction |

**Metric:** MAE in lakhs is the primary metric (the average error a buyer or seller sees). RMSE, R² and MAPE are also reported.
Validation uses an 80/20 hold-out set plus 5-fold cross-validation.

## Results
| Model | Test MAE | Test R² | 5-fold CV MAE (± std) | CV R² |
|---|---|---|---|---|
| **XGBoost (boosting)** | 14.99 | 0.967 | **17.06 ± 0.98** | **0.938** |
| Stacking | 15.03 | 0.966 | 17.28 ± 1.18 | 0.930 |
| Hybrid RF + ANN (blend) | **14.49** | **0.973** | 17.42 ± 1.50 | 0.922 |
| Random Forest (bagging) | 15.29 | 0.965 | 17.75 ± 1.23 | 0.924 |
| Hybrid RF → ANN (residual) | 15.54 | 0.966 | 18.68 ± 1.36 | 0.922 |

**Key takeaways**
* **XGBoost is the best and most stable model** in cross-validation, and has the smallest model file.
* The **RF + ANN hybrid** has the best hold-out score. However, the ANN can extrapolate wildly on rare supercars, which makes the hybrid less stable across folds.
* **Stacking adds little:** the errors of RF and XGBoost are 92 % correlated.
* **Brand + model, engine size and age** set the price. Real mileage matters within a given model and year, but adds little unique information beyond age.

## Repository structure
```
├── SL_Vehicle_Price_Ensemble_Learning.ipynb   # the whole project: cleaning, training, charts, interpretation (all code inside)
├── app.py                                     # Streamlit dashboard (loads models/ and artifacts/)
├── data/ikman_vehicle_listings.csv            # raw scraped dataset
├── models/*.pkl                               # trained pipelines (preprocessing + model), written by the notebook
├── artifacts/*.csv|json                       # metrics, importances, cleaned data used by the dashboard
├── static/                                    # Archivo font for the dashboard (SIL Open Font License)
├── .streamlit/config.toml                     # dashboard theme
├── requirements.txt
└── VIDEO_SCRIPT.md
```

## Run the dashboard (Windows)
```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## AI Usage Acknowledgment
AI tools were used as supporting tools during development. They helped with:
- Fixing coding errors and debugging issues in the notebook.
- Understanding and correcting some ANN implementation issues, including the RF + ANN hybrid approach.
- Clarifying misunderstood concepts and improving the model implementation.
- Developing the Streamlit dashboard, including its layout, charts, and interactive components.
- Improving code readability and documentation.

