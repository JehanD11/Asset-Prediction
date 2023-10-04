import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Load the stock data
df = pd.read_csv('AAPL.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Extract the closing price as the target variable
y = df['Close'].values

# Extract the features
X = df[['Open', 'High', 'Low', 'Volume']].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a sequential model
model = keras.Sequential()

# Add a dense layer
model.add(keras.layers.Dense(1, input_shape=(X_train.shape[1],)))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Predict the stock prices
y_pred = model.predict(X_test)

model.save('AAPL.h5')

# Plot the predicted stock prices
plt.plot(df['Date'][len(X_train):], y_pred, color='red', label='Predicted')
# Plot the actual stock prices
plt.plot(df['Date'][len(X_train):], y_test, color='blue', label='Actual')
# Add labels and a title
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Predicted vs Actual Stock Prices')
plt.legend()
# Show the plot
plt.show()

def predict_price(date):
    # convert the date string to a datetime object
    date = pd.to_datetime(date, format='%d-%m-%Y')
    # find the index of the date in the dataframe
    index = df[df['Date'] == date].index[0]
    # use the index to retrieve the features for the date
    features = X[index].reshape(1, -1)
    # use the model to predict the stock price for the date
    price = model.predict(features)[0][0]
    return price

date = '10-02-2023'
price = predict_price(date)
print(f'The predicted stock price for {date} is {price:.2f}')