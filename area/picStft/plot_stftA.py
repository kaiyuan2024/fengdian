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
frequencies4, times4, Zxx4 = signal.stft(X_train[2], window='hann', fs=fs, nperseg=64)

params={'font.family':'serif',
        'font.serif':'Times New Roman',
        'font.style':'normal',
        'font.weight':'normal', #or 'blod'
        'font.size':10.5}
plt.rcParams.update(params)

# 绘制 STFT 输出
plt.figure(figsize=(3.94, 2.56))
fontsize = 10.5
plt.pcolormesh(times4, frequencies4, np.abs(Zxx4), shading='gouraud', cmap='coolwarm')
plt.ylabel('f/Hz', fontname='Times New Roman', fontsize=fontsize)
plt.xlabel('t/s', fontname='Times New Roman', fontsize=fontsize)
plt.colorbar()

plt.tight_layout()
plt.savefig("stft_A4320-fs10-coolwarm.png", dpi=300)
plt.show()


def colorbar(transport_gdf):
        fig, ax = plt.subplots(figsize=(6, 1))
        fig.subplots_adjust(bottom=0.5)

        norm = mpl.colors.Normalize(vmin=np.power(10, 2), vmax=np.power(10, 7))

        fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap="GnBu"),
                cax=ax,
                orientation="horizontal",
        )


