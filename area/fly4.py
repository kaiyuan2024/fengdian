import pandas as pd
from sklearn.model_selection import train_test_split
from skmultilearn.adapt import MLkNN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy import signal
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def calculate_metrics(pred, target, threshold=0.5):
    target = target.astype(np.int64)
    pred = np.array(pred > threshold, dtype=float)
    return {
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            }

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

classifier = MLkNN(k=6)  # k为近邻数，可以根据需要调整

# 预测测试集标签
classifier.fit(X_train, y_train.values)
predictions = classifier.predict(X_test)
print(type(np.array(predictions)))
print(type(y_test.values))
print(predictions.shape)
print(y_test.values.shape)

# 计算准确率
accuracy1 = accuracy_score(y_test.values, predictions)
recall_1 = recall_score(y_test.values, predictions,average='micro')
f1_1 = f1_score(y_test.values, predictions,average='micro')
# result = calculate_metrics(np.array(predictions), y_test.values)

# 预测测试集标签
classifier.fit(X_train_scaler, y_train.values)
predictions2 = classifier.predict(X_test_scaler)
# 计算准确率
accuracy2 = accuracy_score(y_test.values, predictions2)
recall_2 = recall_score(y_test.values, predictions2,average='micro')
f1_2 = f1_score(y_test.values, predictions2,average='micro')


print("normal Accuracy {} f1 {} reclass {}".format(accuracy1,f1_1, recall_1))
print("improve Accuracy {} f1 {} reclass {}".format(accuracy2,f1_2, recall_2))
    
