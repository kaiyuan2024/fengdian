import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from alexnet import AlexNet
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from scipy import signal
fs = 10e3
parser = argparse.ArgumentParser(description='Resnet Improve')
parser.add_argument('-I', '--improve', default=True,help=' improve?')
data = pd.read_excel("/data2/zhangyong/workspace/project/0306_mul/dataAll.xlsx")
features = data.iloc[:, :-2]  # 前面的列作为特征
labels = data.iloc[:, -2:]  # 最后两列作为标签
# 使用get_dummies函数进行one-hot编码
one_hot_labels = pd.get_dummies(labels)
X_train, X_test, y_train, y_test = train_test_split(features, one_hot_labels, test_size=0.01, random_state=42)
X_train, X_test = np.array(X_train), np.array(X_test)
y_train, y_test = np.array(y_train), np.array(y_test)
class CustomDataset(Dataset):
    def __init__(self, X, y):
        _  ,_, self.X = signal.stft(X, fs, nperseg=1000)
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32)  # 假设标签是整数类型
        return sample, label
def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }
#check gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#set hyperparameter
EPOCH = 10
pre_epoch = 0
BATCH_SIZE = 128
LR = 0.001
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)
#prepare dataset and preprocessing
batch_size = 32
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size)
#labels in CIFAR10

#define ResNet18


#define loss funtion & optimizer
criterion = nn.BCELoss()



def train(args,save_name):
    best_acc = 0
    if args.improve:
        net = AlexNet(use_att= args.improve).to(device)
        optimizer = optim.Adam(net.parameters(), lr=LR)
    else:
        net = AlexNet().to(device)
        optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    save_epochs = []
    save_f1 = []
    save_pre = []

    save_recalss = []
    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            #prepare dataset
            length = len(trainloader)
            inputs, labels = data
            inputs_expanded = inputs.unsqueeze(2)  # 在第二维上添加一个维度

    # 通过广播将维度复制到所需的尺寸
            inputs_broadcasted = inputs_expanded.repeat(1,1,43,1)
            # inputs_broadcasted = inputs_broadcasted.view(inputs_broadcasted.shape[0], 1, 84, 84)
            # inputs_reshaped = inputs_broadcasted.repeat(1, 3, 1, 1)
            
            inputs_reshaped = inputs_broadcasted.permute(0,3, 1, 2)
            
            # images = tensor.permute(0, 2, 3, 1)  # 将通道维度放到最后
    # 插值缩放图像至目标大小 224x224
            resized_images = F.interpolate(inputs_reshaped, size=(224, 224), mode='bilinear', align_corners=False)
            # 将图像转换回张量，并将通道维度放回原位
            # inputs_reshaped = resized_images.permute(0, 3, 1, 2)
            inputs, labels = resized_images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            #forward & backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            #print ac & loss in each batch
            sum_loss += loss.item()
            # _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # correct += predicted.eq(labels.data).cpu().sum()
            # print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' 
            #       % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
            
        #get the ac with testdataset in each epoch
        # scheduler.step()
        print('Waiting Test...')
        with torch.no_grad():
            model_result = []
            targets = []
            for data in testloader:
                net.eval()
                images, labels = data
                inputs_expanded = images.unsqueeze(2)
                inputs_broadcasted = inputs_expanded.repeat(1,1,43,1)
            # inputs_broadcasted = inputs_broadcasted.view(inputs_broadcasted.shape[0], 1, 84, 84)
            # inputs_reshaped = inputs_broadcasted.repeat(1, 3, 1, 1)
            
                inputs_reshaped = inputs_broadcasted.permute(0,3, 1, 2)
                resized_images = F.interpolate(inputs_reshaped, size=(224, 224), mode='bilinear', align_corners=False)
                # inputs_reshaped = resized_images.permute(0, 3, 1, 2)
                images, labels = resized_images.to(device), labels.to(device)
                outputs = net(images)
                model_result.extend(outputs.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        result = calculate_metrics(np.array(model_result), np.array(targets))
        if result['samples/precision'] > best_acc:
            best_acc = result['samples/precision']
        save_epochs.append(epoch)
        save_f1.append(result['samples/f1'])
        save_pre.append(result['samples/precision'])
        save_recalss.append(result['samples/recall'])
        print("epoch:{:2d} test: "
                "micro f1: {:.3f} "
                "macro f1: {:.3f} "
                "samples f1: {:.3f}".format(epoch, 
                                            result['samples/precision'],
                                            result['samples/recall'],
                                            result['samples/f1']))
    combined_data = np.column_stack((save_epochs, save_f1,save_pre,save_recalss))
    np.savetxt(save_name, combined_data, fmt='%.6f', header='Epoch f1 pre rec', delimiter='\t')

    print('Train has finished, total epoch is {}'.format(best_acc))
if __name__ == "__main__":
    args = parser.parse_args()
    train(args,"alexnet_imporve.txt")
    args.imporve = False
    train(args,"alexnet.txt")