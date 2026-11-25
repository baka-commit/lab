#Experiment 12 - Create a LSTM model

import pandas as pd
import zipfile
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Open the ZIP archive and read the target CSV directly
zip_file_path = "/content/DL_Exp_12_source_Files.zip"
target_csv_name = "Google_Stock_Price_Train.csv"

with zipfile.ZipFile(zip_file_path) as z:
    with z.open(target_csv_name) as f:
        df = pd.read_csv(f)

display(df.head())

# The simulated dataset has columns: Date, Open, High, Low, Volume.
# It does not contain a 'Close' column, and the values are already numeric (no commas).
# We can create a mock 'Close' column or use 'Open' for the target.
df['Close'] = df['Open']
df.head()

#Take only close column
train_set = df.iloc[:, 4:5].values
train_set

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
train_set=sc.fit_transform(train_set)
train_set

len(train_set)

#Price existing tomorrow will be the predictive price for today
x_train = train_set[0:1257]
y_train = train_set[1:1258]
x_train.shape
(1257, 1)
#Reshape x_train into 3D as (1257 rows, 1 col, 1 timestamp)
import numpy as np
x_train = np.reshape(x_train,(1257,1,1))
x_train

#Build the RNN model
model = Sequential(
 [
 LSTM(units=50, return_sequences = True, input_shape = (None,
1)),
 LSTM(units=50, return_sequences = True),
 Dropout(0.25),
 Dense(1, activation = 'sigmoid')
 ]
)
#Compile the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
#Train the model
model.fit(x_train, y_train, batch_size = 10, epochs=30)

#Making predictions
import zipfile
import pandas as pd

zip_file_path = "/content/DL_Exp_12_source_Files.zip"
target_csv_name = "Google_Stock_Price_Test.csv"

with zipfile.ZipFile(zip_file_path) as z:
    with z.open(target_csv_name) as f:
        test_set = pd.read_csv(f)

test_set.head()

test_set = test_set.iloc[:,4:5].values
test_set

#Normalization
inputs = sc.transform(test_set)
inputs

#Given data in the form of 2D, we have to convert it into 3D
inputs = np.reshape(inputs, (20,1,1))
inputs

inputs.shape

#Model prediction
predicted_values = model.predict(inputs)
predicted_values

#The output is in 3D array that contains scaled data. Lets convert it into 2D as:
predicted_values = np.reshape(predicted_values, (20,1))
predicted_values

#Use inverse_transform() method for converting scaled data into normal data.
test_pred = sc.inverse_transform(predicted_values)
test_pred

#Visualize the original test data and predicted data
import matplotlib.pyplot as plt
plt.plot(test_set, color='red', label='Original test data')
plt.plot(test_pred, color='green', label='Predicted data')
plt.xlabel("Time")
plt.ylabel("Google Stok Price")
plt.legend()
plt.show()

#predict the next day stock price
today_price = float(input("Enter today's stock price "))

#Convert into 2D array
arr = np.array(today_price).reshape(-1,1)
arr1 = sc.transform(arr)
#Predict this data in 3D format
arr1 = arr1.reshape(-1,1,1)
result = model.predict(arr1)
print(result)

#Convert again in 2D
result = np.reshape(result, (1,1))
print(result)

tomorrow_price = sc.inverse_transform(result)
print("The tomorrow's price is %.2f" % tomorrow_price[0, 0])