from flask import Flask, render_template, request
import yfinance as yf
from prophet import Prophet
import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Home route with form
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    stock = request.form['stock']
    date_input = request.form['date']

    try:
        # ------------------- Prophet Model -------------------
        df = yf.download(stock, period="2y")
        df = df.reset_index()[['Date', 'Close']]
        df.columns = ['ds', 'y']

        prophet = Prophet()
        prophet.fit(df)

        future = prophet.make_future_dataframe(periods=30)
        forecast = prophet.predict(future)

        # Ensure date exists in forecast
        target_date = pd.to_datetime(date_input)
        if target_date not in forecast['ds'].values:
            raise ValueError("Selected date is not available in the forecast range.")

        target_price_prophet = forecast[forecast['ds'] == target_date]['yhat'].values[0]

        # ------------------- XGBoost Model (Dynamic Training) -------------------
        df_xgb = yf.download(stock, period="2y")
        df_xgb = df_xgb.reset_index()
        df_xgb['day'] = df_xgb['Date'].dt.day
        df_xgb['month'] = df_xgb['Date'].dt.month
        df_xgb['year'] = df_xgb['Date'].dt.year
        df_xgb['dayofweek'] = df_xgb['Date'].dt.dayofweek
        df_xgb['dayofyear'] = df_xgb['Date'].dt.dayofyear

        features_xgb = df_xgb[['day', 'month', 'year', 'dayofweek', 'dayofyear']]
        target_xgb = df_xgb['Close']

        scaler_xgb = StandardScaler()
        scaled_features_xgb = scaler_xgb.fit_transform(features_xgb)

        xgb_model = XGBRegressor()
        xgb_model.fit(scaled_features_xgb, target_xgb)

        # Predict for the input date
        target_features = pd.DataFrame({
            'day': [target_date.day],
            'month': [target_date.month],
            'year': [target_date.year],
            'dayofweek': [target_date.dayofweek],
            'dayofyear': [target_date.dayofyear]
        })
        scaled_input = scaler_xgb.transform(target_features)
        target_price_xgb = xgb_model.predict(scaled_input)[0]

        # Render the result to HTML
        return render_template('index.html',
                               stock=stock,
                               date=date_input,
                               prophet_price=round(target_price_prophet, 2),
                               xgb_price=round(target_price_xgb, 2))

    except Exception as e:
        return render_template('index.html', error=str(e))

# Run without reloader to avoid Jupyter crash
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)