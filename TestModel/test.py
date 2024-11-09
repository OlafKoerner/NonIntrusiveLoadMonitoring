# import libs
from decouple import Config, RepositoryEnv, Csv  # https://github.com/HBNetwork/python-decouple/issues/116
import numpy as np
import pymysql
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, LSTM, Flatten
#import tensorflow_model_optimization as tfmot
#import keras
#from keras.utils import plot_model
#import pydot, graphviz
from random import random

# configuration
device_list = {
    1: {'name': 'espresso-machine', 'minpow': 800},
    2: {'name': 'washing-machine', 'minpow': 500},
    4: {'name': 'dish-washer', 'minpow': 500},
    8: {'name': 'induction-cooker', 'minpow': 800},
    # 16: {'name': 'irrigation-system', 'minpow': 400},
    # 32: {'name': 'oven', 'minpow': 800},
    # 64: {'name': 'microwave', 'minpow': 800},
    # 128: {'name': 'kitchen-light', 'minpow': 200},
    # 256: {'name': 'living-room-light', 'minpow': 200},
    # 512: {'name': 'dining-room-light', 'minpow': 200},
    # 1024: {'name': 'ground-floor-light', 'minpow': 200},
    # 2048: {'name': 'upper-floor-light', 'minpow': 200},
}
window_length = 20
model = Sequential()
epochs = 200
[3]
config = Config(RepositoryEnv(".env"))
conn = pymysql.connect(
    host=config('myhost'),
    user=config('myuser'),
    password=config('mypassword'),
    database=config('mydatabase'),
    cursorclass=pymysql.cursors.DictCursor)
[5]
cur = conn.cursor()

xx = np.array([])
yy = np.array([])

for key in device_list.keys():
    # read from mysql db
    # cur.execute("SELECT * FROM data WHERE device = " + str(key) + " LIMIT 10000")
    cur.execute("SELECT * FROM data WHERE device = " + str(key) + " AND timestamp > 1698221020000 LIMIT 10000")
    # get all rows where device is active
    data_list = cur.fetchall()
    # storage for values for current active device
    x = np.array([])

    # size_x = x.size
    active = False
    for row in data_list:
        if not active:
            if row['value'] > device_list[key]['minpow']:
                active = True
                x = np.append(x, 200 + (random() - 0.5) * abs(100))
                x = np.append(x, row['value'])
        else:
            if row['value'] > device_list[key]['minpow']:
                x = np.append(x, row['value'])
            else:
                active = False
                x = np.append(x, 200 + (random() - 0.5) * abs(100))

    # init batch targets for this device
    batch_target_values = np.zeros(len(device_list))
    batch_target_values[int(np.log2(key))] = 1.

    # generate batches with values and targets

    i = 0 + window_length
    while i < x.size:
        xx = np.append(xx, x[i - window_length: i])
        yy = np.append(yy, batch_target_values)
        i = i + window_length
    xx = xx.reshape((xx.size // window_length, window_length))
    yy = yy.reshape((yy.size // len(device_list), len(device_list)))

conn.close()
[6]
for i in range(xx[0].size):
    plt.plot(xx[i])
print(xx[0].size)