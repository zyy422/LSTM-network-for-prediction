import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import load_model
import matplotlib.pyplot  as plt
import math
from sklearn.metrics import mean_squared_error


def read_data():
    data = pd.read_excel('./212-Site_25-Hanwha-Solar/2017.xlsx', sheet_name=2)
    test = pd.read_excel('./212-Site_25-Hanwha-Solar/2018.xlsx', sheet_name=2)
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


# def delete_night_point(real_y, predict_y):
#     index_0 = np.where(real_y == 0)
#     for i in index_0:
#         predict_y[i] = 0
#     return predict_y
#
#
# def delete_zero(real_y, predict_y):
#     amount_day = int(real_y.shape[0] / 288)
#     for i in range(amount_day):
#         upper_bound = np.arange(0, 64, 1, dtype=np.int) + 288*i
#         down_bound = np.arange(208, 268, 1, dtype=np.int) + 288*i
#         np.delete(real_y, upper_bound, axis=0)
#         np.delete(predict_y, down_bound, axis=0)
#     return real_y, predict_y
# def adopt_daytime(real_y, predict_y):
#     amount_day = int(real_y.shape[0]/288)
#     buff_real_y = []
#     buff_predict_y = []
#     for i in range(amount_day):
#         buff_real_y.append(real_y[288*i+64:288*i+208])
#         buff_predict_y.append(predict_y[288*i+64:288*i+208])
#     return buff_real_y, buff_predict_y


if __name__ == "__main__":
    TIMESTEP = 1440     # 单位：min
    DELAY = 200
    training, testing = read_data()
    print("This is the "+str(DELAY/5)+" time")
    train_x, train_y, test_x, test_y, scale_1, scale_2 = normalization(training, testing)
    train_dataset = np.hstack((train_y, train_x))
    test_dataset = np.hstack((test_y, test_x))
    train_x, train_y = create_interval_dataset(train_dataset, TIMESTEP, DELAY)
    test_x, test_y = create_interval_dataset(test_dataset, TIMESTEP, DELAY)
    real_y = scale_2.inverse_transform(test_y)
    model_name = "./lstm_200min/winter_sheet_2.h5"
    model = load_model(model_name)
    predict_y = model.predict(test_x, batch_size=256)
    predict_y = predict_y.reshape(-1, 1)
    predict_y = scale_2.inverse_transform(predict_y)
    rmse = math.sqrt(mean_squared_error(real_y, predict_y))
    predict_y[predict_y < 0] = 0
    print('rmse:', rmse)
    pearson_coeff=np.corrcoef(real_y.T, predict_y.T)
    print('pearson_coeff:', pearson_coeff)
    plt.scatter(predict_y, real_y, s=5, c='k')
    plt.ylim((0, 7.5))
    plt.text(0, 6.5, 'Pearson Coefficient: 0.9448', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.text(0, 6, 'RMSE: 0.6053 (kW)', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.text(0, 7, 'Time Horizon: 200 (min)',fontdict={'family': 'Times New Roman', 'size': 16})
    plt.xlabel('Forecast Output (kW)', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.ylabel('Actual Output (kW)', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.title('Fall', fontdict={'family': 'Times New Roman', 'size': 16})
    plt.savefig('./lstm_200min/Fall.png')
    plt.show()
    # print(rmse)
    # delete_night_point(real_y, predict_y)
    # real_y, predict_y = delete_zero(real_y, predict_y)
    # real_y, predict_y = adopt_daytime(real_y, predict_y)
    # model_1 = np.hstack((real_y, predict_y))
    # np.savetxt('model_result_3.csv', model_1, delimiter=',')
    # plt.plot(predict_y[10000:12000], label='LSTM', color='r', linestyle='dashed')
    # plt.plot(real_y[10000:12000], label='Actual', color='k')
    # plt.xlabel('Minute')
    # plt.ylabel('PV power (kW)')
    # plt.ylim((0, 6))
    # plt.title('Summer')
    # plt.legend()
    # plt.savefig('Summer.png')
    # plt.show()


