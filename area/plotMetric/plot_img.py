import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = np.loadtxt('../0306_mul/alexnet_imporve.txt', skiprows=1)  # 跳过文件头部
data2 = np.loadtxt('../0306_mul/alexnet.txt', skiprows=1)  # 跳过文件头部
# 提取各列数据

epochs = data[:, 0]
atten_f1 = data[:, 1]
atten_pre = data[:, 2]
atten_recall = data[:, 3]

f1 = data2[:, 1]
pre = data2[:, 2]
recall = data2[:, 3]
# 绘制图表
plt.figure(figsize=(10, 5))
# plt.plot(epochs, train_losses, label='Training Loss')

# plt.plot(epochs, val_losses, label='Val Loss')
plt.plot(epochs, atten_f1, label='Attention Training',color='b')
plt.plot(epochs, f1, label='Normal Training',color='r')
plt.xlabel('Epoch')
plt.ylabel('f1')
plt.title('Training f1')
plt.legend()
plt.grid(True)
plt.savefig("f1.jpg")
plt.clf()
plt.plot(epochs, atten_pre, label='Attention Training',color='b')
plt.plot(epochs, pre, label='Normal Training',color='r')
plt.xlabel('Epoch')
plt.ylabel('acc')
plt.title('Training acc')
plt.legend()
plt.grid(True)
plt.savefig("acc.jpg")
plt.clf()
plt.plot(epochs, atten_recall, label='Attention Training',color='b')
plt.plot(epochs, recall, label='Normal Training',color='r')
plt.xlabel('Epoch')
plt.ylabel('recall')
plt.title('Training recall')
plt.legend()
plt.grid(True)
plt.savefig("recall.jpg")