import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_excel("../0306_mul/dataAll.xlsx")
features = data.iloc[:, :-2]  # 前面的列作为特征
labels = data.iloc[:, -2:]  # 最后两列作为标签
one_hot_labels = pd.get_dummies(labels)
fs = 1e2
X_train, X_test, y_train, y_test = train_test_split(features, one_hot_labels, test_size=0.2, random_state=42)
# 对特征数据进行 STFT
X_train = np.array(X_train)

# 计算四种不同参数设置下的 STFT 结果
frequencies1, times1, Zxx1 = signal.stft(X_train[1], window='boxcar', fs=fs, nperseg=500)
frequencies2, times2, Zxx2 = signal.stft(X_train[1], window='boxcar', fs=fs, nperseg=1000)
frequencies3, times3, Zxx3 = signal.stft(X_train[1], window='hann', fs=fs, nperseg=100)
frequencies4, times4, Zxx4 = signal.stft(X_train[1], window='hann', fs=fs, nperseg=200)

# 绘制 STFT 输出
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.pcolormesh(times1, frequencies1, np.abs(Zxx1), shading='gouraud')
plt.title('STFT Magnitude - boxcar, nperseg=64', fontname='SimSun', fontsize=10.5)
plt.ylabel('Frequency [Hz]', fontname='SimSun', fontsize=10.5)
plt.xlabel('Time [sec]', fontname='SimSun', fontsize=10.5)
plt.colorbar(label='Magnitude')

plt.subplot(2, 2, 2)
plt.pcolormesh(times2, frequencies2, np.abs(Zxx2), shading='gouraud')
plt.title('STFT Magnitude - boxcar, nperseg=32', fontname='SimSun', fontsize=10.5)
plt.ylabel('Frequency [Hz]', fontname='SimSun', fontsize=10.5)
plt.xlabel('Time [sec]', fontname='SimSun', fontsize=10.5)
plt.colorbar(label='Magnitude')

plt.subplot(2, 2, 3)
plt.pcolormesh(times3, frequencies3, np.abs(Zxx3), shading='gouraud')
plt.title('STFT Magnitude - hann, nperseg=64', fontname='SimSun', fontsize=10.5)
plt.ylabel('Frequency [Hz]', fontname='SimSun', fontsize=10.5)
plt.xlabel('Time [sec]', fontname='SimSun', fontsize=10.5)
plt.colorbar(label='Magnitude')

plt.subplot(2, 2, 4)
plt.pcolormesh(times4, frequencies4, np.abs(Zxx4), shading='gouraud')
plt.title('STFT Magnitude - hann, nperseg=32', fontname='SimSun', fontsize=10.5)
plt.ylabel('Frequency [Hz]', fontname='SimSun', fontsize=10.5)
plt.xlabel('Time [sec]', fontname='SimSun', fontsize=10.5)
plt.colorbar(label='Magnitude')

plt.tight_layout()
plt.savefig("stft_comparison_font-fs100-longwin.png", dpi=300)
plt.show()
