import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# getting the stock data
company = 'AAPL'
stock_df = web.DataReader(company, data_source='yahoo', start='2012-01-01', end='2021-04-25')
# looking at the data
print(stock_df)

# let's plot the closing values for training data first
plt.figure(figsize=(16, 10))
plt.title('Close Price Over Years for {}'.format(company))
plt.plot(stock_df['Close'], color='green')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD', fontsize=18)
plt.show()

# creating a new dataframe using the close values
df = stock_df.filter(['Close'])
# now converting to numpy
data = df.values
# next we get the number of rows on which we want to train our model
train_data_len = math.ceil(len(data)*.8)
print(train_data_len)

# scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
print(scaled_data)

# creating dataset for training
train_data = scaled_data[0:train_data_len, :]

# next we split this data into x_train and y_train
x_train = []
y_train = []
# building the datasets
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# converting the above two datasets to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# reshaping the training dataset for LSTM model
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

# here we build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# compiling the above model
model.compile(optimizer='adam', loss='mean_squared_error')

# training the model
model.fit(x_train, y_train, epochs=25, batch_size=32)

# creating the test dataset
test_data = scaled_data[train_data_len-60:, :]
# creating x_test and y_test datasets
x_test = []
y_test = data[train_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# again we convert the test data to numpy array
x_test = np.array(x_test)

# reshaping the data again
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# next we get the models predicted price values
predictions = model.predict(x_test)
# inverse transforming (unscaling values)
predictions = scaler.inverse_transform(predictions)
print(y_test)
# now we test performance - get the RMSE
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
print('The RMSE value of this model is:{}'.format(rmse))

# plotting the results
## getting data ready
train = df[:train_data_len]
actual = df[train_data_len:]
actual['Predictions'] = predictions

## visualizing
plt.figure(figsize=(16, 10))
plt.xlabel('Date', fontsize=18)
plt.title('LSTM Model with RMSE of {} for Stock Price Prediction for company: {}'.format(rmse, company))
plt.ylabel('Close Price in USD', fontsize=18)
#plt.plot(train['Close'])
plt.plot(actual[['Close', 'Predictions']])
plt.legend(['Actual', 'Predictions'], loc='upper left')
plt.show()