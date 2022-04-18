from datetime import datetime
import numpy
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

st.title('Stock Trend Prediction') #Title for streamlit page

user_input = st.text_input("Enter Stock Ticker", "AAPL") #take input in text-box

#start = '2010-01-01'
start = str(yf.Ticker(user_input).history(period='max').reset_index()['Date'][0])[:10] #Extract start date of the stock from Yahoo
#end = '2014-02-16'
end = datetime.today().strftime("%Y-%m-%d") #Today's date as End date

df = data.DataReader(user_input, 'yahoo', start, end) #Take data of stock from yahoo in DataFrame
df = df.reset_index() #Reset Index of Dataframe to remove date as index and put 1,2,3,4...

st.subheader(str(start)+" To "+str(end)) #Put subheading as Start - End
st.write(df.describe()) #Put the table of Data Frame in StreamLit

st.subheader('Closing Price vs Time Chart') #Sub-Heading
fig = plt.figure(figsize=(12,6)) #create and set dimensions for the figure
plt.plot(df.Close) #plot data frame close points
st.pyplot(fig) #display figure on streamlit

st.subheader('Closing Price vs Time Chart with 100 MA') #Sub-Heading
ma100 = df.Close.rolling(100).mean() #For Rolling window calculation - it will give mean at every interval of 100. If not rolling then it will give only one mean
fig = plt.figure(figsize=(12, 6)) #create and set dimensions for the figure
plt.plot(ma100, 'r') #plot MA100 graph
plt.plot(df.Close) #plot data frame close points
st.pyplot(fig) #display figure on streamlit

st.subheader('Closing Price vs Time Chart with 100 MA & 200 MA')
ma100 = df.Close.rolling(100).mean() #For Rolling window calculation - it will give mean at every interval of 100. If not rolling then it will give only one mean
ma200 = df.Close.rolling(200).mean() #For Rolling window calculation - it will give mean at every interval of 200. If not rolling then it will give only one mean
fig = plt.figure(figsize=(12, 6)) #create and set dimensions for the figure
plt.plot(ma100, 'r')  #plot MA100 graph
plt.plot(ma200, 'g')  #plot MA200 graph
plt.plot(df.Close) #plot data frame close points
st.pyplot(fig) #display figure on streamlit

df1 = df.reset_index()['Close'] #create a new dataframe with the indexes reset to number other than date

#Split Data into Training and Testing
#Model prediction requires data to be in range 0 - 1
#MinMax scaler converts data in the range of 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1)) #define function to convert data in 0 and 1
df1 = scaler.fit_transform(numpy.array(df1).reshape(-1, 1)) # Convert the data of the dataframe in 0-1
train_data, test_data = df1[0:int(len(df1)*0.70), :], df1[int(len(df1)*0.70):len(df1), :1] # Divide data into training and Testing in ratio 70:30

#Load my model
model = load_model('new_model.h5') #Load previously trained model

#Function to create dataset
def create_dataset(dataset, time_step=1):
    dataX,dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return numpy.array(dataX), numpy.array(dataY)

#Predict data 101th day data, taking previous 100 days into consideration

time_step=100
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1) #reshape in 1D array
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1) #reshape in 1D array
train_predict = model.predict(x_train) #Predict x_train with the trained model
test_predict = model.predict(x_test) #Predict x_test with trained model
train_predict = scaler.inverse_transform(train_predict) #Convert the 0-1 data back to original form
test_predict = scaler.inverse_transform(test_predict) #Convet 0-1 data back to original form

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
#plt.show()
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


