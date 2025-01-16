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

# Load in data & reshape and scale from range 0 to 1
path = "C:/Walking_Data/"
df_Y = pd.DataFrame()
df_X = pd.DataFrame()
for filename in os.listdir(path):
    if filename.endswith("R_Fmedial.csv"): % Ground Truth Data
        data = pd.read_csv(filename,header = None)
        df_Y = pd.concat([df_Y,data],axis = 1)
    elif filename.endswith("R_Sensor.csv"): % Sensor Data
        data = pd.read_csv(filename,header = None)
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

# Set groups of each subject 
groups = np.concatenate([np.ones(12)*1, np.ones(11)*2, np.ones(14)*3, np.ones(17)*4,
                         np.ones(7)*5, np.ones(11)*6, np.ones(16)*7, np.ones(17)*8])

#Build model 
model = keras.Sequential()
model.add(layers.Conv2D(160, (15, 1), activation = 'relu', input_shape = (101,15,1)))
model.add(layers.Dropout(rate = 0.1))
model.add(layers.Conv2D(80, (15,1), activation = 'relu'))
model.add(layers.Dropout(rate = 0.1))
model.add(layers.Conv2D(16, (15,1), activation = 'relu'))
model.add(layers.Dropout(rate = 0.1))
model.add(layers.Flatten())
model.add(layers.Dense(101,activation = 'relu'))
model.add(layers.Dropout(rate = 0))
model.add(layers.Dense(101))
model.add(layers.Dropout(rate = 0))

optimizer = keras.optimizers.Adam(learning_rate=1e-3)

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

   
    mod = model.fit(x_train, y_train,  epochs = 100,
                    callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'loss',patience=15)], batch_size = 23, verbose= 0) # batch size should be number of samples
   # summarize history for loss
    plt.plot(mod.history['loss'])
    plt.plot(mod.history['mse']) #RAISE ERROR
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    scores = model.evaluate(x_test, y_test) # calculate mse score
    mse.extend(scores) # store mse
    
    pred = model.predict(x_test) # produce current model prediction of x
    result.extend(pred) #store prediction

    expected = y_test # expected result of prediction
    val.extend(expected)

# Calculate mean and standard deviation of mse for results
avg_mse = np.mean(mse)
std_mse = np.std(mse)
print("Mean mse:", avg_mse, "Std:", std_mse)
