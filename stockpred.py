


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

StockCode = input("Enter Stock Code: ").upper()
print (f"Downloading {StockCode} data")
data = yf.download(StockCode, start="2020-01-01", end= "2025-01-01")
data = data [["Close"]]



data["Target"] = data["Close"].shift(-1)
data = data.dropna()

X = data[["Close"]]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, test_size=0.2
)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)

plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(preds, label="Predicted")
plt.title(f"{StockCode} Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

latest_price = data["Close"].iloc[-1]
tomorrow_pred = model.predict([[latest_price]])[0]

print(f"Latest close price for {StockCode}: {latest_price:.2f} USD")
print(f"Predicted next close for {StockCode}: {tomorrow_pred:.2f} USD")