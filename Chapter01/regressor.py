#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'GonnaZero'

import numpy as np

x = []
y = []
filename = 'data_multivar.txt'
with open(filename, 'r') as f:
    for line in f.readlines():
        data = [float(i) for i in line.split(',')]
        xt, yt = data[:-1], data[-1]
        x.append(xt)
        y.append(yt)

num_training = int(0.8 * len(x))
num_test = len(x) - num_training

# Training data
x_train = np.array(x[:num_training])
y_train = np.array(y[:num_training])

# Test data
x_test = np.array(x[num_training:])
y_test = np.array(y[num_training:])

# Create linear regression object
# 建立岭回归器
from sklearn import linear_model

linear_regressor = linear_model.LinearRegression()
ridge_regressor = linear_model.Ridge(alpha=0.01, fit_intercept=True, max_iter=10000)

# Train the model using the traing sets
linear_regressor.fit(x_train, y_train)
ridge_regressor.fit(x_train, y_train)

# Predict the output
y_test_pred = linear_regressor.predict(x_test)
y_test_pred_ridge = ridge_regressor.predict(x_test)

# Measure performance
import  sklearn.metrics as sm
print("LINEAR:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

print("\nRIDGE:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_ridge), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred_ridge), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred_ridge), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred_ridge), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred_ridge), 2))

# Polynomial regression
# 创建多项回归器
from sklearn.preprocessing import PolynomialFeatures

polynomial = PolynomialFeatures(degree=10) #degree:多项式最高次数
x_train_transformed = polynomial.fit_transform(x_train)
# 用数据检测多项式是否预测准确
datapoint = np.array([[0.39, 2.78, 7.11]]) #少一个中括号形成二维数组
print(datapoint.reshape(-1, 1).shape) # 此处reshape(-1, 1)不可以随便用
print(datapoint.reshape(-1, 1).ndim, "变换之后的维度")
print(datapoint.shape, "行列数")
print(datapoint.ndim, "维度")
poly_datapoint = polynomial.fit_transform(datapoint)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(x_train_transformed, y_train)
print("\nLInear regression: ", linear_regressor.predict(datapoint)[0])
print("\nPolynomial regression: ", poly_linear_model.predict(poly_datapoint)[0])
print("\n实际值: -8.07")

#将多项式的次数加到10
#polynomial = PolynomialFeatures(degree=10)
#print("\nPolynomial regression: ", poly_linear_model.predict(poly_datapoint)[0])


