#!/usr/bin/env python
# coding: utf-8

# In[1]:


#STEP 1 : IMPORTING ALL THE REQUIRED LIBRARIES
import numpy as np #imports the NumPy library and assigns it the alias np
import matplotlib.pyplot as plt #imports the pyplot module from the matplotlib library and assigns it the alias plt
import pandas as pd # imports the Pandas library and assigns it the alias pd
from sklearn.preprocessing import StandardScaler # imports the StandardScaler class from the sklearn.preprocessing module
from sklearn.model_selection import train_test_split  #imports the train_test_split function from the sklearn.model_selection module
import tensorflow as tf #imports the TensorFlow library and assigns it the alias tf
from tensorflow.keras.models import Sequential #imports the Sequential class from the tensorflow.keras.models module
from tensorflow.keras.layers import Dense, LSTM, Dropout #imports the three layers i.e., Dense, LSTM and Dropout from the tensorflow.keras.layers modules
from tensorflow.keras.callbacks import EarlyStopping  #imports the EarlyStopping callback from the tensorflow.keras.callbacks module
from sklearn.metrics import mean_squared_error #imports the mean_squared_error function from the sklearn.metrics module
import datetime #Python library for working with dates and times
import matplotlib.dates as mdates #functionality for formatting and manipulating dates on plots


# In[2]:


#STEP 2 : READING THE DOWNLOADED DATASET USING THE PANDAS LIBRARY
netflix = pd.read_csv("Downloads/NFLX.csv", index_col='Date') # read a csv file with Date column as the index of the DataFrame


# In[3]:


#STEP 3 : DISPLAY THE DATA
netflix


# In[4]:


netflix.info()  # getting information about the structure and properties of the Dataset netflix


# In[5]:


#STEP 4 : PLOTTING THE GRAPH OF HIGH AND LOW VALUES OF THE STOCK

plt.figure(figsize=(25,15)) #create a new figure for a plot with a specified size
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) #configuring the x-axis to display dates in the format YYYY-MM-DD
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=100)) #configuring the x-axis to have major ticks at intervals of 100 days
netflix_dates = [datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in netflix.index.values] #creates a list by converting the date strings in the index of a DataFrame into datetime objects


plt.plot(netflix_dates, netflix['High'], label = 'High') #plots the High values from the netflix
plt.plot(netflix_dates, netflix['Low'], label = 'Low') #plots the Low values from the netflix
plt.legend() #display the legend
plt.gcf().autofmt_xdate() #format the date labels on the x-axis in a way that optimally fits the available space
plt.show() #display the Matplotlib plot


# In[6]:


#STEP 5 : CREATING X & Y VARIABLES 
X = netflix.iloc[:, 0:3] #creates a new DataFrame X by selecting columns from the existing netflix DataFrame
Y = netflix['Close'] #creates a new Series Y by extracting the 'Close' column from the DataFrame netflix
X #prints X
Y #prints Y 


# In[7]:


X.iloc[:,0:]  #select all rows and columns starting from index 0 and continuing to the end


# In[8]:


X.values.shape #get the shape of X


# In[9]:


#STEP 6 : STANDARDISING THE VALUES OF X USING STANDARDSALER LIBRARY
columns = X.columns #assigns it the column names of X
rows = X.index #assigns it the row labels of X
sc = StandardScaler() #creates an instance of the StandardScaler class 

X = sc.fit_transform(X.values) #standardize the values
X = pd.DataFrame(columns = columns, data = X, index = rows) #creates new dataframe after standardizing


# In[10]:


X #display updated X


# In[11]:


#STEP 7 : CREATE A FUNCTION TP SPLIT THE DATASET FOR TIMESERIES FORECASTING
def split_function(netflix_features, netflix_labels, n_steps): #defines a function
    X, y = [], [] # initializes two empty lists, X and y
    for i in range(len(netflix_features) - n_steps + 1): #beginning of a for loop
        X.append(netflix_features[i:i + n_steps]) #appends a subsequence of features to the list X
        y.append(netflix_labels[i + n_steps - 1]) #appends a subsequence of features to the list y
    return np.array(X), np.array(y) #returns the accumulated input sequences and corresponding labels 


# In[12]:


x1, y1 = split_function(X.values,Y, 10) #create input sequences and corresponding labels


# In[13]:


x1.shape #shape of x1


# In[14]:


#STEP 8 : SPLITTING THE DATA INTO TRAINING AND TESTING DATASETS

P = int(np.ceil(len(x1)*0.8)) #index at which data split will take
Q =  netflix.index  #index of netflix dataframe

x_train, x_test = x1[:P], x1[P:] #splitting the input sequences x1
y_train, y_test = y1[:P], y1[P:] #splits the labels
x_train_date, x_test_date = Q[:P], Q[P:] #creating two sets of date indices


# In[15]:


print(x_train.shape, x_test.shape, x_train_date.shape, x1.shape) #priniting shape of testing and training data


# In[16]:


tf.keras.backend.clear_session() #clear the computational graph and reset the state of the TensorFlow session


# In[26]:


#STEP 9 : DEFINING THE ARCHITECTURE OF THR NEURAL NETWORK
model = Sequential() # initializes a Sequential modeL
model.add(LSTM(100, input_shape=(x_train.shape[1],x_train.shape[2]  ), activation='relu', return_sequences=True)) #adds an LSTM layer to the Sequential model
model.add(Dense(1)) #adds a Dense layer with a single unit
model.compile(loss='mean_squared_error', optimizer='adam') #compiles the Keras Sequential model
model.summary() #provides a summary of the architecture 


# In[27]:


#STEP 10 : TRAINING THE MODEL 
TR_NTFX = model.fit(x_train, y_train, epochs = 500, batch_size= 2, verbose=2, shuffle=False)


# In[28]:


#STEP 11 : PLOTTING THE GRAPH OF LOSS VS EPOCHS

plt.figure(figsize = (25,15)) # creates a new figure for a Matplotlib plot
plt.plot([i for i in range(0,500)], TR_NTFX.history['loss']) #create a line plot
plt.title('Learning Curve') # to give title to graph
plt.xlabel('number of epochs') #label to x axis
plt.ylabel('Loss') #label to y axis
plt.legend(['Training Loss']) #adds a legend to the plot
plt.show() #display the graph


# In[29]:


#STEP 12 : PREDICTING THE VALUES USING PREDICT FUNCTION
pred = model.predict(x_test)


# In[30]:


y_pred = pred.reshape(501,-1) # reshapes the pred array


# In[31]:


y_pred = y_pred.mean(axis=1) #calculates the mean along axis 1


# In[32]:


y_pred.shape #retrieve the shape of y_pred


# In[33]:


error =  mean_squared_error(y_test,y_pred, squared = False) #calculates the Root Mean Squared Error between the true labels and the predicted values
print('RMSE score on test dataset:',error) #Printing root mean square error


# In[34]:


#STEP 13 : PLOTTING THE GRAPH OF ACTUAL AND PREDICTED VALUES TO SEE THE PERFORMANCE OF THE MODEL
fig, ax = plt.subplots(figsize=(16,8)) #creates a new Matplotlib figure and axes for a plot
ax.set_facecolor('#425666') #set color of axis
ax.plot(y_test, color='green', label='Original price') #create a line plot on the subplot represented by the ax
plt.plot(y_pred, color='red', label='Predicted price') #plotting the predicted prices
plt.legend()


# In[ ]:




