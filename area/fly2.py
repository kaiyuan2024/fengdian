import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from skmultilearn.adapt import MLkNN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy import signal

fs=10e3
def FFT(signal):
    return np.fft.fft(signal)

# read data
data = pd.read_excel("dataAll.xlsx")
features = data.iloc[:, :-2]  # 前面的列作为特征
labels = data.iloc[:, -2:]  # 最后两列作为标签
one_hot_labels = pd.get_dummies(labels)
X_train, X_test, y_train, y_test = train_test_split(features, one_hot_labels, test_size=0.2, random_state=42)
print(type(X_train.values))
X_train_fft, X_test_fft = np.real(FFT(X_train)),np.real(FFT(X_test))
X_train, X_test = signal.stft(X_train, fs, nperseg=84)[-1], signal.stft(X_test, fs, nperseg=84)[-1]
X_train, X_test = np.real(X_train.reshape(X_train.shape[0], -1)), np.real(X_test.reshape(X_test.shape[0], -1))
# scaler = StandardScaler()
# X_train_scaler = scaler.fit_transform(X_train)
# X_test_scaler = scaler.transform(X_test)

for k in range(1,15):
    classifier = MLkNN(k=k)  # k为近邻数，可以根据需要调整

    # 预测测试集标签
    classifier.fit(X_train, y_train.values)
    predictions = classifier.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test.values, predictions)
    print("Accuracy:", accuracy)

    # 预测测试集标签
    classifier.fit(X_train_fft, y_train.values)
    predictions = classifier.predict(X_test_fft)

    # 计算准确率
    accuracy2 = accuracy_score(y_test.values, predictions)

    if accuracy2 > accuracy:
        print("best k is {}".format(k))
    print("Accuracy2:", accuracy2)


# for k in range(1,10):
#     classifier = MLkNN(k=k)  # k为近邻数，可以根据需要调整
#     classifier.fit(X_train, y_train.values)

# # 预测测试集标签
#     predictions = classifier.predict(X_test)

#     # 计算准确率
#     accuracy = accuracy_score(y_test.values, predictions)
#     print(accuracy)
