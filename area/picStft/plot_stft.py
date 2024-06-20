import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_excel("../0306_mul/dataAll.xlsx")
features = data.iloc[:, :-2]  # 前面的列作为特征
labels = data.iloc[:, -2:]  # 最后两列作为标签
one_hot_labels = pd.get_dummies(labels)
fs = 10e3
X_train, X_test, y_train, y_test = train_test_split(features, one_hot_labels, test_size=0.2, random_state=42)
# 对特征数据进行 STFT
X_train = np.array(X_train)
frequencies, times, Zxx = signal.stft(X_train[3], window='boxcar', fs=10e2, nperseg=64)

# 绘制 STFT 输出
plt.pcolormesh(times, frequencies, np.abs(Zxx), shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Magnitude')
plt.savefig("boxcar64-fs100.png", dpi=300)
plt.show()
