# Developed by: Sriram S
# Reg no: 212222240105
# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('/content/WMT.csv')  

# Convert 'Date' to datetime format and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Check for stationarity using the Augmented Dickey-Fuller (ADF) test on 'Volume'
result = adfuller(data['Volume']) 
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Split data into training and testing sets (80% training, 20% testing)
train_data = data.iloc[:int(0.8 * len(data))]
test_data = data.iloc[int(0.8 * len(data)):]

# Define the lag order for the AutoRegressive model (adjust lag based on ACF/PACF plots)
lag_order = 13
model = AutoReg(train_data['Volume'], lags=lag_order)
model_fit = model.fit()

# Plot Autocorrelation Function (ACF) for 'Volume'
plt.figure(figsize=(10, 6))
plot_acf(data['Volume'], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF) - Volume')
plt.show()

# Plot Partial Autocorrelation Function (PACF) for 'Volume'
plt.figure(figsize=(10, 6))
plot_pacf(data['Volume'], lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF) - Volume')
plt.show()

# Make predictions on the test set
predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

# Calculate Mean Squared Error (MSE) for the test set predictions
mse = mean_squared_error(test_data['Volume'], predictions)
print('Mean Squared Error (MSE):', mse)

# Plot Test Data vs Predictions for 'Volume'
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data['Volume'], label='Test Data - Volume', color='blue', linewidth=2)
plt.plot(test_data.index, predictions, label='Predictions - Volume', color='orange', linestyle='--', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('AR Model Predictions vs Test Data (Volume)')
plt.legend()
plt.grid(True)
plt.show()
```
### OUTPUT:

### GIVEN DATA

![image](https://github.com/user-attachments/assets/bcc013ab-210a-46ff-ad83-e8c6044d027a)

### PACF - ACF

![image](https://github.com/user-attachments/assets/7d2f0f47-e0c7-4ba6-bf13-a6d71316f870)

### PREDICTION

![image](https://github.com/user-attachments/assets/fdbe4568-d3c7-43da-a6bb-cfe759249fc9)

### FINIAL PREDICTION

![image](https://github.com/user-attachments/assets/6ba79d9f-ac92-4608-8d8a-dfd36a331fa0)

### RESULT:
ThuS, successful implemention of the auto regression function using python is done.
