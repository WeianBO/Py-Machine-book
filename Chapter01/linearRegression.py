#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'GonnaZero'

import sys
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import pickle
from sklearn.preprocessing import PolynomialFeatures



'''创建线性回归器'''
# filename = sys.argv[1] 命令行输入文件名时使用
#读取数据
x = []
y = []
with open('./data_singlevar.txt', 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        x.append(xt)
        y.append(yt)

#建立机器学习模型
num_traing = int(0.8 * len(x))
num_test = len(x) - num_traing
#训练数据
x_traing = np.array(x[:num_traing]).reshape((num_traing, 1))
y_traing = np.array(y[:num_traing])
#测试数据
x_test = np.array(x[num_traing:]).reshape((num_test, 1))
y_test = np.array(y[num_traing:])

#创建回归器对象
#创建线性回归对象
linear_regressor = linear_model.LinearRegression()
#用训练数据集训练模型
linear_regressor.fit(x_traing, y_traing)

#可视化图形观察拟合

y_train_pred = linear_regressor.predict(x_traing)
plt.figure(figsize = (10, 8), dpi = 100) #指定创建几张图，一个plt.figure()创建一张
plt.subplot(211) # 2行1列的第1行
plt.scatter(x_traing, y_traing, color='green') #scatter:点
plt.plot(x_traing, y_train_pred, color='black', linewidth=4)
plt.title("Trainingg, data")

#用模型对测试数据集进行预测
y_test_pred = linear_regressor.predict(x_test)
plt.subplot(212)
plt.scatter(x_test, y_test, color='green')
plt.plot(x_test, y_test_pred, color='black', linewidth=4)
plt.title("Test data")
#plt.show()

'''计算回归准确性'''

# 回归器拟合指标
print("Mean absolute error = ", round(sm.mean_absolute_error(y_test, y_test_pred), 2)) # 平均绝对误差
print("Mean squared error = ", round(sm.mean_squared_error(y_test, y_test_pred), 2)) # 均方误差
print("Median absoule error = ", round(sm.median_absolute_error(y_test, y_test_pred), 2)) # 中位数绝对误差
print("Explained variance score = ", round(sm.explained_variance_score(y_test, y_test_pred), 2)) # 解释方差分
print("R2 score = ", round(sm.r2_score(y_test, y_test_pred), 2)) # R方得分

'''保存模型数据'''

output_model_file = 'saved_model.pkl'
with open(output_model_file, 'wb') as f:
    pickle.dump(linear_regressor, f)

#加载使用
with open(output_model_file, 'rb') as f:
    model_linregr = pickle.load(f)
y_test_pred_new = model_linregr.predict(x_test)
print("\nNew mean absolute error = ", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))



