#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'GonnaZero'

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn import datasets
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

'''估算房屋价格'''


def main():
    # 使用带AdaBoost算法的决策树回归器

    # 调用房屋价格数据接口 每个数据点含有13个输入参数
    housing_data = datasets.load_boston()
    # .data 获取输入参数， .target获取对应价格
    x, y = shuffle(housing_data.data, housing_data.target, random_state=7)
    # 分成训练集和测试集
    num_training = int(0.8 * len(x))
    x_train, y_train = x[:num_training], y[:num_training]
    x_test, y_test = x[num_training:], y[num_training:]

    # 拟合一个决策回归模型
    dt_regressor = DecisionTreeRegressor(max_depth=4)
    dt_regressor.fit(x_train, y_train)
    # 用带AdaBoost算法的决策回归模型拟合
    ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
    ab_regressor.fit(x_train, y_train)

    # 评价决策树回归器的效果
    y_pred_dt = dt_regressor.predict(x_test)
    mse = mean_squared_error(y_test, y_pred_dt)
    evs = explained_variance_score(y_test, y_pred_dt)
    print("\n###  Decision Tree performance ####")
    print("Mean squared error =", round(mse, 2))
    print("Explained variance score = ", round(evs, 2))

    # AdaBoost算法改善后的效果
    y_pred_ab = ab_regressor.predict(x_test)
    mse = mean_squared_error(y_test, y_pred_ab)
    evs = explained_variance_score(y_test, y_pred_ab)
    print("\n###  Decision Tree performance ####")
    print("Mean squared error =", round(mse, 2))
    print("Explained variance score = ", round(evs, 2))


    # 画出特征的相对重要性
    #plt.figure(figsize=(10, 8), dpi=100)  # 指定尺寸和分辨率
    plot_feature_importances(dt_regressor.feature_importances_, 'Decision Tree regressor', housing_data.feature_names,1)
    plot_feature_importances(ab_regressor.feature_importances_, 'AdaBoost regressor', housing_data.feature_names, 2)
    #plt.show()  # 改变代码，画在一张图中，方便比较

# 定义函数，画出条形图
def plot_feature_importances(feature_importances, title, feature_names, i):
    # 将重要性值标准化
    feature_importances = 100.0 * (feature_importances / max(feature_importances))
    # 将得分从高到低排序
    index_sorter = np.flipud(np.argsort(feature_importances))
    # 让x坐标轴上的标签居中显示
    pos = np.arange(index_sorter.shape[0]) + 0.5 # shape[0] 读取第一维度长度
    # 画条形图
    plt.figure(i, figsize = (10, 8), dpi = 100) # 指定尺寸和分辨率
    plt.subplot(210+i)
    plt.bar(pos, feature_importances[index_sorter], align='center') #条形图
    plt.xticks(pos, feature_names[index_sorter])
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    main()










