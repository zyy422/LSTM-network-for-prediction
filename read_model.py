import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import load_model
# import matplotlib.pyplot  as plt
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error


def read_data():
    data = pd.read_excel('./212-Site_25-Hanwha-Solar/2017.xlsx', sheet_name=1)
    test = pd.read_excel('./212-Site_25-Hanwha-Solar/2018.xlsx', sheet_name=1)
    data = data.fillna(0)
    data = data.values
    test = test.fillna(0)
    test = test.values
    return data, test


def normalization(training, testing):
    scale_1 = MinMaxScaler(feature_range=(-1, 1))
    scale_2 = MinMaxScaler(feature_range=(-1, 1))
    train_y = training[:, 1:2]
    test_y = testing[:, 1:2]
    train_y[train_y < 0] = 0
    test_y[test_y < 0] = 0
    train_x = scale_1.fit_transform(training[:, 2:])
    train_y = scale_2.fit_transform(train_y)
    test_x = scale_1.transform(testing[:, 2:])
    test_y = scale_2.transform(test_y)
    train_x, train_y, test_x, test_y = np.array(train_x, dtype=np.float32), np.array(train_y, dtype=np.float32), \
        np.array(test_x, dtype=np.float32), np.array(test_y, dtype=np.float32)
    return train_x, train_y, test_x, test_y, scale_1, scale_2


def create_interval_dataset(dataset, look_back, delay): # lookback --> 需要回顾的时间, dealy --> 预测delay时间后的值
    look_back = np.int(look_back/5)
    delay = np.int(delay/5)
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - delay):   # 从第二天开始
        dataX.append(dataset[i:i+look_back, 1:])
        dataY.append(dataset[i+look_back+delay, 0])
    dataX, dataY = np.array(dataX), np.array(dataY)
    dataY = dataY.reshape(-1, 1)
    return dataX, dataY


if __name__ == "__main__":
    TIMESTEP = 1440     # 单位：min
    buff_rmse = []
    training, testing = read_data()
    for DELAY in range(50, 150, 5):
        print("This is the "+str(DELAY/5)+" time")
        train_x, train_y, test_x, test_y, scale_1, scale_2 = normalization(training, testing)
        train_dataset = np.hstack((train_y, train_x))
        test_dataset = np.hstack((test_y, test_x))
        train_x, train_y = create_interval_dataset(train_dataset, TIMESTEP, DELAY)
        test_x, test_y = create_interval_dataset(test_dataset, TIMESTEP, DELAY)
        real_y = scale_2.inverse_transform(test_y)
        model_name = "./model_1/short_forecast_"+str(DELAY)+"min.h5"
        model = load_model(model_name)
        predict_y = model.predict(test_x, batch_size=256)
        predict_y = predict_y.reshape(-1, 1)
        predict_y = scale_2.inverse_transform(predict_y)
        rmse = math.sqrt(mean_squared_error(real_y, predict_y))
        buff_rmse.append(rmse)
    print(buff_rmse)
