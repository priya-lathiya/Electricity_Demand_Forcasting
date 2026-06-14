# Electricity Demand Forecasting using XGBoost

This is a machine learning project where I built a model to forecast electricity demand (power consumption) using historical time-series data. I made this project to practice data cleaning, feature engineering, and building a regression model for time-series data.

## About the Project

Predicting electricity demand is useful because it helps in better planning of power generation and avoiding wastage. In this project, I used household power consumption data and applied **time-series feature engineering** along with the **XGBoost** algorithm to predict future electricity usage.

## Dataset

- Dataset: [Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235) (UCI Machine Learning Repository)
- Original data: ~2 million rows recorded at 1-minute intervals (from Dec 2006 to Nov 2010)
- Target column: `Global_active_power` (household power consumption in kW)

## Tech Stack / Libraries Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn (for visualization)
- XGBoost (for the model)
- Scikit-learn (for evaluation metrics & scaling)

## Project Workflow

**Step 1: Load the Dataset**
- Fetched the dataset directly from the UCI repository using `ucimlrepo`

**Step 2: Data Preprocessing & Cleaning**
- Combined `Date` and `Time` columns into a single `Datetime` index
- Converted all columns to proper numeric types
- Handled missing values (forward-fill for short gaps, dropped the rest)
- Capped outliers in power consumption using the 1st–99th percentile range
- Resampled the data from 1-minute intervals to **hourly** data for easier modeling

**Step 3: Exploratory Data Analysis (EDA)**
- Visualized electricity consumption patterns by hour of day, day of week, and month
- Found that usage is higher during evenings/mornings, slightly higher on weekends, and much higher in winter months

**Step 4: Feature Engineering**
- Created **lag features** (previous 1h, 2h, 3h, 6h, 12h, 24h, 48h, and 1 week)
- Added **rolling mean & rolling standard deviation** features (3h, 6h, 24h, 168h windows)
- Added **cyclical encoding** (sin/cos) for hour and month, so the model understands that hour 23 and hour 0 are close to each other

**Step 5: Build the XGBoost Model**
- Split the data chronologically into 80% train and 20% test (no shuffling, since it's time-series data)
- Trained an `XGBRegressor` model with early stopping to avoid overfitting

**Step 6: Model Evaluation**
- Evaluated the model using RMSE, MAE, R² Score, and MAPE
- Plotted the training loss curve and feature importance graph

**Step 7: Visualize Predictions vs Actual**
- Compared predicted values with actual values on the test set
- Did a monthly performance breakdown

## Results

| Metric | Train | Test |
|--------|-------|------|
| RMSE   | 0.0135 kW | 0.0343 kW |
| MAE    | 0.0096 kW | 0.0192 kW |
| R²     | 0.9998 | 0.9978 |
| MAPE   | 1.37% | 2.19% |

**Model Accuracy: ~97.8%** | **Variance Explained (R²): ~99.8%**

The lag features and rolling mean features turned out to be the most important for the model, which makes sense since electricity usage strongly depends on its recent past values.

## How to Run This Project

1. Clone this repository
git clone https://github.com/priya-lathiya/Electricity_Demand_Forcasting.git

2. Install the required libraries
pip install -r requirements.txt

3. Open `Electricity_Demand_Forcasting.ipynb` in Jupyter Notebook or Google Colab
4. Run all the cells step by step

## About Me

This is one of my learning projects as I explore data science and machine learning. Feedback and suggestions are always welcome!

## Note

This project is for learning purposes and can be improved further by trying other models (like LSTM) or adding external features such as weather data.
