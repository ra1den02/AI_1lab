import numpy as np
from sklearn import preprocessing
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from utilities import visualize_classifier
input_data = np.array([[-1.3, 3.9, 4.5],
                        [-5.3, -4.2, -1.3],
                        [5.2, -6.5, -1.1],
                        [-5.2, 2.6, -2.2]])
data_binarized = preprocessing.Binarizer(threshold=3.0).transform(input_data)
print("\n Binarized data:\n", data_binarized)
print("\nBEFORE: ")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))
data_scaled = preprocessing.scale(input_data)
print("\nAFTER: ")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

# Масштабування MinМax
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nМin max scaled data:\n", data_scaled_minmax)
# Нормалізація даних
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nl1 normalized data:\n", data_normalized_l1)
print("\nl2 normalized data:\n", data_normalized_l2)
# Надання позначок вхідних даних
# input_labels = ['red', 'Ыасk', 'red', 'green', 'Ьlack',
# 'yellow', 'white']
#
# # Створення кодувальника та встановлення відповідності
# # між мітками та числами
# encoder = preprocessing.LabelEncoder()
# encoder.fit(input_labels)
# # Виведення відображення
# print("\nLabel mapping:")
# for i, item in enumerate(encoder.classes_): print(item, '-->', i)
#
# # перетворення міток за допомогою кодувальника
# test_labels = ['green', 'red', 'Ыасk']
# encoded_values = encoder.transform(test_labels)
# print("\nLabels =", test_labels)
# print("Encoded values =", list(encoded_values))
# # Декодування набору чисел за допомогою декодера
# encoded_values = [3, 0, 4, 1]
# decoded_list = encoder.inverse_transform(encoded_values)
# print("\nEncoded values =", encoded_values)
# print("Decoded labels =", list(decoded_list))