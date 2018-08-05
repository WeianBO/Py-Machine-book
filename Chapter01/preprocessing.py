#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'GonnaZero'

import numpy as np
from sklearn import preprocessing

data = np.array([[3, -1.5, 2, -5.4], [0, 4, -0.3, 2.1], [1, 3.3, -1.9, -4.3]])
print("原始数据\n", data)

#均值移除
data_standardized = preprocessing.scale(data)
print("\nMean = " , data_standardized.mean(axis=0)) #axis=0 列
print("Std deviation = " , data_standardized.std(axis=0))

#范围缩放
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaler = data_scaler.fit_transform(data)
print("\nMin max scaled data = ", data_scaler)

#归一化
data_normalized = preprocessing.normalize(data, norm='l1') #是 l（字母） 不是 1（数字）
print("\n normalized data = ", data_normalized)

#二值化
data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
print("\nBinarized data = ", data_binarized)

print("######################################")
#独热编码
encoder = preprocessing.OneHotEncoder()
encoder.fit([[0, 2, 1, 12], [1, 3, 5, 3], [2, 3, 2, 12], [1, 2, 4, 3]])
encoder_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
print("\nEncoder vector = ", encoder_vector)
