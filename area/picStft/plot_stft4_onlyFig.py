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
# plt.figure()
plt.pcolormesh(times1, frequencies1, np.abs(Zxx1), shading='gouraud', cmap='coolwarm')
# plt.show()
plt.axis('off')  # 去坐标轴
plt.xticks([])  # 去x轴刻度
plt.yticks([])  # 去y轴刻度
# 保存图片 去白边
plt.savefig("stft_onlyFig-fs10-boxcar-32-coolwarm.png", dpi=300, bbox_inches='tight', pad_inches=0)

# 绘制 STFT 输出
# plt.figure()
plt.pcolormesh(times2, frequencies2, np.abs(Zxx2), shading='gouraud', cmap='coolwarm')
# plt.show()
plt.axis('off')  # 去坐标轴
plt.xticks([])  # 去x轴刻度
plt.yticks([])  # 去y轴刻度
# 保存图片 去白边
plt.savefig("stft_onlyFig-fs10-hann-32-coolwarm.png", dpi=300, bbox_inches='tight', pad_inches=0)

# 绘制 STFT 输出
# plt.figure()
plt.pcolormesh(times3, frequencies3, np.abs(Zxx3), shading='gouraud', cmap='coolwarm')
# plt.show()
plt.axis('off')  # 去坐标轴
plt.xticks([])  # 去x轴刻度
plt.yticks([])  # 去y轴刻度
# 保存图片 去白边
plt.savefig("stft_onlyFig-fs10-boxcar-64-coolwarm.png", dpi=300, bbox_inches='tight', pad_inches=0)

# 绘制 STFT 输出
# plt.figure()
plt.pcolormesh(times4, frequencies4, np.abs(Zxx4), shading='gouraud', cmap='coolwarm')
# plt.show()
plt.axis('off')  # 去坐标轴
plt.xticks([])  # 去x轴刻度
plt.yticks([])  # 去y轴刻度
# 保存图片 去白边
plt.savefig("stft_onlyFig-fs10-hann-64-coolwarm.png", dpi=300, bbox_inches='tight', pad_inches=0)

# plt.figure()
# plt.colorbar()