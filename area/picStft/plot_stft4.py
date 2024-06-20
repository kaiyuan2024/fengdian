import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_excel("../0306_mul/dataAll.xlsx")
features = data.iloc[:, :-2]  # 前面的列作为特征
labels = data.iloc[:, -2:]  # 最后两列作为标签
one_hot_labels = pd.get_dummies(labels)
fs = 10
X_train, X_test, y_train, y_test = train_test_split(features, one_hot_labels, test_size=0.2, random_state=42)
# 对特征数据进行 STFT
X_train = np.array(X_train)

# 计算四种不同参数设置下的 STFT 结果
frequencies1, times1, Zxx1 = signal.stft(X_train[1], window='boxcar', fs=fs, nperseg=32)
frequencies2, times2, Zxx2 = signal.stft(X_train[1], window='hann', fs=fs, nperseg=32)
frequencies3, times3, Zxx3 = signal.stft(X_train[1], window='boxcar', fs=fs, nperseg=64)
frequencies4, times4, Zxx4 = signal.stft(X_train[1], window='hann', fs=fs, nperseg=64)

# 绘制 STFT 输出
plt.figure(figsize=(6, 4.6))
fontsize = 10.5
plt.subplot(2, 2, 1)
plt.pcolormesh(times1, frequencies1, np.abs(Zxx1), shading='gouraud')
plt.title('L=32 - Rectangular Window', fontsize=fontsize)
plt.ylabel('f/Hz', fontsize=fontsize)
plt.xlabel('t/s', fontsize=fontsize)
ax = plt.gca()
# ax.tick_params(axis='x', labelsize=10)  # 设置x轴刻度标签字体大小为10
# ax.tick_params(axis='y', labelsize=10)  # 设置y轴刻度标签字体大小为10
plt.colorbar()

plt.subplot(2, 2, 2)
plt.pcolormesh(times2, frequencies2, np.abs(Zxx2), shading='gouraud')
plt.title('L=32 - Hann Window', fontsize=fontsize)
plt.ylabel('f/Hz', fontsize=fontsize)
plt.xlabel('t/s', fontsize=fontsize)
plt.colorbar()

plt.subplot(2, 2, 3)
plt.pcolormesh(times3, frequencies3, np.abs(Zxx3), shading='gouraud')
plt.title('L=64 - Rectangular Window', fontsize=fontsize)
plt.ylabel('f/Hz', fontsize=fontsize)
plt.xlabel('t/s', fontsize=fontsize)
plt.colorbar()

plt.subplot(2, 2, 4)
plt.pcolormesh(times4, frequencies4, np.abs(Zxx4), shading='gouraud')
plt.title('L=64 - Hann Window', fontsize=fontsize)
plt.ylabel('f/Hz', fontsize=fontsize)
plt.xlabel('t/s', fontsize=fontsize)
plt.colorbar()

plt.tight_layout()
plt.savefig("stft_comparison-fs10.png", dpi=300)
plt.show()
