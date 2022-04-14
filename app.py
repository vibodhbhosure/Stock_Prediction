from datetime import datetime
import numpy
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

st.title('Stock Trend Prediction')

user_input = st.text_input("Enter Stock Ticker", "AAPL")

#start = '2010-01-01'
start = str(yf.Ticker(user_input).history(period='max').reset_index()['Date'][0])[:10]
#end = '2014-02-16'
end = datetime.today().strftime("%Y-%m-%d")

df = data.DataReader(user_input, 'yahoo', start, end)
df = df.reset_index()

st.subheader(str(start)+" To "+str(end))
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 MA & 200 MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close)
st.pyplot(fig)

df1 = df.reset_index()['Close']

#Split Data into Training and Testing
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(numpy.array(df1).reshape(-1, 1))
train_data, test_data = df1[0:int(len(df1)*0.70), :], df1[int(len(df1)*0.70):len(df1), :1]

#Load my model
model = load_model('new_model.h5')


def create_dataset(dataset, time_step=1):
    dataX,dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return numpy.array(dataX), numpy.array(dataY)

time_step=100
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

loopback = 100
st.subheader('Testing of Prediction')
fig6 = plt.figure(figsize=(12, 6))
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[loopback:len(train_predict)+loopback, :] = train_predict
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(loopback*2)+1:len(df1)-1, :] = test_predict
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
st.pyplot(fig6)


x_input=test_data[len(test_data)-100:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()


lst_output = []
n_steps = 100
i = 0
npt = int(st.text_input("How many days of data to predict?", "10"))
while (i < npt):

    if (len(temp_input) > 100):
        # print(temp_input)
        x_input = numpy.array(temp_input[1:])
        print("{} day input {}".format(i, x_input))
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        # print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        # print(temp_input)
        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i = i + 1

day_new = numpy.arange(1,101)
day_pred = numpy.arange(101,101+npt)

st.subheader('Limited Prediction')
fig7 = plt.figure(figsize=(12, 6))
df3 = df1.tolist()
df3.extend(lst_output)
plt.plot(day_new, scaler.inverse_transform(df1[len(df1)-100:]),'g')
plt.plot(day_pred, scaler.inverse_transform(lst_output),'r')
df3 = df1.tolist()
df3.extend(lst_output)
plt.plot(df3[12000:])
st.pyplot(fig7)


st.subheader('Overall Prediction')
fig8 = plt.figure(figsize=(12, 6))
df3 = df1.tolist()
df3.extend(lst_output)
#plt.plot(day_new,scaler.inverse_transform(df1[len(df1)-100:]))
#plt.plot(day_pred,scaler.inverse_transform(lst_output))
df3 = df1.tolist()
df3.extend(lst_output)
plt.plot(df3[:],'g')
st.pyplot(fig8)


