import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from copy import deepcopy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import os


def str_to_datetime(date_str):
    return datetime.datetime.strptime(date_str, '%Y-%m-%d')


def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    first_date = str_to_datetime(first_date_str)
    last_date = str_to_datetime(last_date_str)

    target_date = first_date
    dates = []
    X, Y = [], []

    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n + 1)

        if len(df_subset) != n + 1:
            st.error(f'Error: Window of size {
                     n} is too large for date {target_date}')
            return

        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_week = dataframe.loc[target_date:target_date +
                                  datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year, month, day = next_date_str.split('-')
        next_date = datetime.datetime(
            day=int(day), month=int(month), year=int(year))

        if last_time:
            break

        target_date = next_date

        if target_date == last_date:
            last_time = True

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates

    X = np.array(X)
    for i in range(0, n):
        ret_df[f'Target-{n - i}'] = X[:, i]

    ret_df['Target'] = Y

    return ret_df


def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:, 0]

    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

    Y = df_as_np[:, -1]

    return dates, X.astype(np.float32), Y.astype(np.float32)


st.title("Stock Prediction Visualization")

image_path = 'hand_invest.jpg'
if os.path.exists(image_path):
    st.image(image_path, caption='Investments Made Wiser!', width=300)
else:
    st.error(f"Image file '{
             image_path}' not found. Please check the file path.")

stock_csv = st.file_uploader("Upload a CSV file", type=["csv"])
start_date = st.text_input("Enter the start date (YYYY-MM-DD)")
end_date = st.text_input("Enter the end date (YYYY-MM-DD)")

if stock_csv and start_date and end_date:
    df = pd.read_csv(stock_csv)
    st.success("CSV file loaded successfully.")
    df = df[['Date', 'Close']]
    df['Date'] = df['Date'].apply(str_to_datetime)
    df.index = df.pop('Date')

    windowed_df = df_to_windowed_df(df, start_date, end_date, n=3)
    if windowed_df is not None:
        dates, X, y = windowed_df_to_date_X_y(windowed_df)
        q_80 = int(len(dates) * .8)
        q_90 = int(len(dates) * .9)

        dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
        dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
        dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

        # Training, validation, and testing data plot
        plt.figure(figsize=(10, 6))
        plt.plot(dates_train, y_train, label='Train')
        plt.plot(dates_val, y_val, label='Validation')
        plt.plot(dates_test, y_test, label='Test')
        plt.legend()
        plt.title("Training, Validation, and Testing Data")
        st.pyplot(plt.gcf())

        # Training predictions
        model = Sequential([
            Input((3, 1)),
            LSTM(64),
            Dense(32, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001),
                      metrics=['mean_absolute_error'])
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

        train_predictions = model.predict(X_train).flatten()
        plt.figure(figsize=(10, 6))
        plt.plot(dates_train, train_predictions, label='Training Predictions')
        plt.plot(dates_train, y_train, label='Training Observations')
        plt.legend()
        plt.title("Training Predictions vs Observations")
        st.pyplot(plt.gcf())

        # Validation predictions
        val_predictions = model.predict(X_val).flatten()
        plt.figure(figsize=(10, 6))
        plt.plot(dates_val, val_predictions, label='Validation Predictions')
        plt.plot(dates_val, y_val, label='Validation Observations')
        plt.legend()
        plt.title("Validation Predictions vs Observations")
        st.pyplot(plt.gcf())

        # Testing predictions
        test_predictions = model.predict(X_test).flatten()
        plt.figure(figsize=(10, 6))
        plt.plot(dates_test, test_predictions, label='Testing Predictions')
        plt.plot(dates_test, y_test, label='Testing Observations')
        plt.legend()
        plt.title("Testing Predictions vs Observations")
        st.pyplot(plt.gcf())

        # Recursive predictions
        model.save('stock_prediction_model.keras')
        recursive_predictions = []
        recursive_dates = np.concatenate([dates_val, dates_test])

        for target_date in recursive_dates:
            last_window = deepcopy(X_train[-1])
            next_prediction = model.predict(np.array([last_window])).flatten()
            recursive_predictions.append(next_prediction)
            last_window[-1] = next_prediction

        plt.figure(figsize=(10, 6))
        plt.plot(dates_train, train_predictions, label='Training Predictions')
        plt.plot(dates_train, y_train, label='Training Observations')
        plt.plot(dates_val, val_predictions, label='Validation Predictions')
        plt.plot(dates_val, y_val, label='Validation Observations')
        plt.plot(dates_test, test_predictions, label='Testing Predictions')
        plt.plot(dates_test, y_test, label='Testing Observations')
        plt.plot(recursive_dates, recursive_predictions,
                 label='Recursive Predictions')
        plt.legend()
        plt.title("All Predictions with Recursive Predictions")
        st.pyplot(plt.gcf())
