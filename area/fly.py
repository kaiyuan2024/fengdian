import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from skmultilearn.adapt import MLkNN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy import signal

fs = 10e3

data = pd.read_excel("dataAll.xlsx")
features = data.iloc[:, :-2]  # 前面的列作为特征
labels = data.iloc[:, -2:]  # 最后两列作为标签
one_hot_labels = pd.get_dummies(labels)
X_train, X_test, y_train, y_test = train_test_split(features, one_hot_labels, test_size=0.2, random_state=42)
print(type(X_train.values))
X_train, X_test = signal.stft(X_train, fs, nperseg=1000)[-1], signal.stft(X_test, fs, nperseg=84)[-1]
X_train, X_test = np.real(X_train.reshape(X_train.shape[0], -1)), np.real(X_test.reshape(X_test.shape[0], -1))

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

def zca_whitening(inputs):
    sigma = np.dot(inputs, inputs.T)/(inputs.shape[1]-1)##Matlab协方差计算的源码需要除以[样本数-1]
    U,S,V = np.linalg.svd(sigma)#奇异分解
    epsilon = 0.1#白化的时候，防止除数为0
    ZCAMatrix = np.dot(np.dot(U,np.diag(1.0/np.sqrt(S+epsilon))),U.T)##S不能使用diag，矩阵会将epsilon广播。
    return np.dot(ZCAMatrix,inputs)

X_train_white = zca_whitening((X_train_scaler))
X_test_white = zca_whitening((X_test_scaler))



for k in range(1,15):
    classifier = MLkNN(k=k)  # k为近邻数，可以根据需要调整

    # 预测测试集标签
    classifier.fit(X_train, y_train.values)
    predictions = classifier.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test.values, predictions)
    print("Accuracy:", accuracy)

    # 预测测试集标签
    classifier.fit(X_train_white, y_train.values)
    predictions = classifier.predict(X_test_white)

    # 计算准确率
    accuracy2 = accuracy_score(y_test.values, predictions)

    if accuracy2 > accuracy:
        print("best k is {}".format(k))
    print("Accuracy2:", accuracy2)
