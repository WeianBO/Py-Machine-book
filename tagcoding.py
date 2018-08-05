#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'GonnaZero'

from sklearn import preprocessing

#定义一个编码器
label_encoder = preprocessing.LabelEncoder()
#创建一些标记
input_classes = ['audi', 'ford', 'audi', 'toyota', 'ford', 'bmw']
#为标记编码
label_encoder.fit(input_classes)
print("\nClass mapping:")
for i, item in enumerate(label_encoder.classes_):
    print(item, '-->', i)
#标记
labels = ['toyota', 'ford', 'audi']
encoded_labels = label_encoder.transform(labels)
print("\nLabels =", labels)
print("Encoded labels =", list(encoded_labels))

#数字反转回单词检查结果正确性
encoded_labels = [2, 1, 0, 3, 1]
decoded_lables = label_encoder.inverse_transform(encoded_labels)
print("\nEncoded labels =", encoded_labels)
print("Decoded labels =", list(decoded_lables))


