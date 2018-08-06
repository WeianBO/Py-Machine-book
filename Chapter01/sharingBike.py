#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'GonnaZero'

import csv
from sklearn.ensemble import RandomForestRegressor
from house import plot_feature_importances  # 导入house文件中的函数
import numpy as np
from sklearn.utils import shuffle

'''随机森林回归器'''
def main():

    # 读取并打乱数据
    x, y, feature_names = load_dataset('bike_day.csv')
    x, y = shuffle(x, y, random_state=7)

    # 将数据分成测试集和训练集
    num_training = int(0.9 * len(x))
    x_train, y_train = x[:num_training], y[:num_training]
    x_test, y_test = x[num_training:], y[num_training:]

    # 创建并训练回归器
    rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10, min_samples_split=2)
    rf_regressor.fit(x_train, y_train)

    # 评价训练效果
    from sklearn.metrics import mean_squared_error, explained_variance_score
    y_pred = rf_regressor.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    print("\n###### Random Forest regressor performance ###")
    print("Mean squared error =", round(mse, 2))  # round:第一个参数是一个浮点数，第二个参数是保留的小数位数
    print("Explained variance score = ", round(evs, 2))

    #调用house中定义的函数，画出特征重要性
    plot_feature_importances(rf_regressor.feature_importances_, 'Random Forestregressor', feature_names, 1)


# 定义一个数据集加载函数
def load_dataset(filename):
    file_reader = csv.reader(open(filename, 'r'), delimiter=',')
    x, y = [], []
    for row in file_reader:
        # 本数据前两列和14，15列无关列，不用考虑
        x.append(row[2:13])
        y.append(row[-1])
    # 提取特征名称
    feature_names = np.array(x[0])
    #将第一行特征名称移除，仅保留数值
    return np.array(x[1:]).astype(np.float32), np.array(y[1:]).astype(np.float32), feature_names


if __name__ == '__main__':
    main()




