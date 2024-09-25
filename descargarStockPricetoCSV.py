import yfinance as yf
import pandas as pd

ticker = "YPF"

data = yf.download(ticker, start= "2004-09-23", end= "2024-09-23")

print(data.head(5))

data.to_csv(f'{ticker}.csv')