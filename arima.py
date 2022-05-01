import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
data = pd.read_excel('output1.xlsx')
data = data[:]
# print(data)
data.index = pd.to_datetime(data.Date)

# print(data.index)

BEA = data['BEA']
x = BEA.head()
BEA.plot()
plt.show()
# print(x)
# quit()
plt.figure(figsize=(20,9))
# plt.title('BEA Plot')
plt.ylabel('BEA')
# BEA.plot()
# plt.xticks(rotation=45)
# plt.show()
BEA_diff1 = BEA.diff(1)
BEA_diff2 = BEA_diff1.diff(1)
BEA_diff3 = BEA_diff2.diff(1)
BEA_diff4 = BEA_diff3.diff(1)
BEA.plot()
plt.show()
plt.figure(figsize=(20,9))
BEA_diff3.plot()
plt.show()
BEA_diff1 = BEA_diff1.fillna(0)
BEA_diff2 = BEA_diff1.fillna(0)
BEA_diff3 = BEA_diff1.fillna(0)
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(BEA_diff1, use_vlines=True, lags=30)
plt.show()

model = ARIMA(BEA, order=(2,2,4))
result = model.fit()
# result.summary()
result.conf_int()
# result.forecast(500)
# plt.figure(figsize=(16,9))
fig = result.plot_predict()
plt.show()
