# common imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.integrate import simps
from scipy.stats import spearmanr
import kerastuner as kt
from kerastuner.tuners import Hyperband, BayesianOptimization, RandomSearch
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import LeaveOneGroupOut
logo = LeaveOneGroupOut()
import os

print(tf.__version__)

#### Load in data, remove Nan, reshape and scale from range 0 to 1
path = "C:/Running_Data/"
df_Y = pd.DataFrame()
df_X = pd.DataFrame()
for filename in os.listdir(path):
    if filename.endswith("R_Fmedial.csv"):
        data = pd.read_csv(filename,header = None).fillna(0)
        df_Y = pd.concat([df_Y,data],axis = 1)
    elif filename.endswith("R_Sensor.csv"):
        data = pd.read_csv(filename,header = None).fillna(0)
        df_X = pd.concat([df_X,data],axis = 1)
    else: 
        continue

# MinMaxScaler - scale data from range of 0 to 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit(df_X))
X_scale = scaler.transform(df_X)
print(X_scale.shape)

# reshape data (samples, timesteps, features)
samples = np.int64(X_scale.shape[1]/15)
np_X = np.array(X_scale)
np_Y = np.array(df_Y)
np_x = np_X.reshape(X_scale.shape[1],101,1)
np_x = np.moveaxis(np_X.reshape(samples,101,-1,1),1,1)
#np_x = np.moveaxis(np_X.reshape(101,samples,-1),0,1)
np_y = np_Y.transpose(-1,0)
print(np_x.shape,np_y.shape)

##### Set groups of each subject and build model
groups = np.concatenate([np.ones(12)*1, np.ones(11)*2, np.ones(14)*3, np.ones(17)*4,
                         np.ones(7)*5, np.ones(11)*6, np.ones(16)*7, np.ones(17)*8])
len(groups)

# Build model 
model = keras.Sequential()
model.add(layers.Conv2D(16, (15, 1), activation = 'relu', input_shape = (101,15,1)))
# model.add(layers.MaxPooling2D(1,2))
model.add(layers.Conv2D(64, (15,1), activation = 'relu'))
# model.add(layers.MaxPooling2D(1,2))
model.add(layers.Conv2D(48, (15,1), activation = 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(101,activation = 'relu'))
model.add(layers.Dropout(rate = 0.1))
model.add(layers.Dense(101))
model.add(layers.Dropout(rate = 0.1))

# Compile model
model.compile(
    loss = tf.keras.losses.MeanAbsoluteError(),
    optimizer="adam",
    metrics=["mse"],)

model.summary()

# Create variable to store mse and prediction result
mse = []
result = []
val = []


# Split data into train and test based on groups 
for train, test in logo.split(np_x, np_y, groups = groups):
    print("TRAIN:",train.shape,"TEST:",test.shape)
    x_train, x_test = np_x[train], np_x[test]
    y_train, y_test = np_y[train], np_y[test]
    print("xtrain:",x_train.shape, "xtest:",x_test.shape, 
      "ytrain:",y_train.shape,"ytest:",y_test.shape)

    mod = model.fit(x_train, y_train, batch_size = 23, epochs = 100) # batch size should be number of samples
   
    scores = model.evaluate(x_test, y_test) # calculate mse score
    mse.extend(scores) # store mse
    
    pred = model.predict(x_test) # produce current model prediction of x
    result.extend(pred) #store prediction

    expected = y_test # expected result of prediction
    val.extend(expected)

#### Calculate mean and standard deviation of results
# Calculate mean and standard deviation of mse from each epoch
avg_mse = np.mean(mse)
std_mse = np.std(mse)
print("Mean mse:", avg_mse, "Std:", std_mse)

# Calculate mean and standard deviation of predictions (acutal results)
result = np.array(result)
avg_p = np.mean(result, axis=0)
avg_p = avg_p.flatten()
std_p = np.std(result, axis=0)
std_p = std_p.flatten()

# Calculate mean and standard deviation of test data (expected results)
val = np.array(val)
avg_t = np.mean(val, axis = 0)
avg_t = avg_t.flatten()
std_t = np.std(val, axis =0)
std_t = std_t.flatten()
print("Test Data STD:", np.mean(std_t))
