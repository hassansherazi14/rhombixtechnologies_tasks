# Apple Stock Price Prediction (LSTM)

LSTM-based deep learning model to predict Apple (AAPL) stock closing prices using historical data.

A stock price prediction project using a Long Short-Term Memory (LSTM) neural network. Historical AAPL closing prices are fetched via `yfinance`, preprocessed into time-series sequences, and used to train an LSTM model that forecasts future closing prices. Includes data visualization, model evaluation (RMSE, MAE, MAPE), and next-day price prediction.

## What's inside

- `AAPL_Stock_Prediction_LSTM.ipynb` — full Jupyter notebook: data download, EDA, preprocessing, LSTM model, training, evaluation, and next-day prediction.

## How it works

1. Downloads AAPL historical closing prices (2015–present) using `yfinance`.
2. Scales prices with `MinMaxScaler` and builds 60-day input sequences.
3. Splits data chronologically into train/test sets (no shuffling — this is time-series data).
4. Trains a 2-layer LSTM network with dropout regularization.
5. Evaluates performance using RMSE, MAE, and MAPE.
6. Plots actual vs. predicted prices.
7. Predicts the next day's closing price.

## Results

| Metric | Value |
|--------|-------|
| RMSE   | 13.48 |
| MAE    | 10.84 |
| MAPE   | 4.34% (~95.7% prediction accuracy) |

The model tracks AAPL's daily closing price movement closely, with an average error of about 4.34% on unseen test data.

## Requirements

```
pip install yfinance tensorflow scikit-learn matplotlib pandas numpy
```

## Usage

Open `AAPL_Stock_Prediction_LSTM.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab, and run all cells top to bottom. An internet connection is required to download stock data.

## Disclaimer

This project is for educational purposes only. Stock prices depend on many real-world factors beyond historical price patterns, and this model should not be used for actual trading or investment decisions.

