import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sktime.transformations.panel.rocket import MiniRocket,MiniRocketMultivariate,MiniRocketMultivariateVariable
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from sklearn import metrics
# 假设 X_train, y_train 分别是你的训练数据特征和标签
l_df = []
num_1 = 0
num_k = 3e3
txt1 = "mr_rccv3-3e3k.txt"
txtmat = "mr_rccv3-3e3k-mat.txt"
for i in range(1, 14):
    l_df.append(pd.read_csv(f'WECC240_U/case{i}BusVolMag.csv', index_col=0))
for i in range(1, 14):
    df_test = l_df[i - 1]
    df_train = pd.concat(l_df[:i - 1] + l_df[i:], axis=0)
    # print(df_train.shape)
    # (n_instances, n_variables, n_timepoints)
    X_train = df_train.iloc[:, :-1]
    n_instances, n_timepoints = X_train.shape
    X_train = np.reshape(X_train.values, (n_instances, 1, n_timepoints))
    y_train = df_train.iloc[:, -1]
    X_test = df_test.iloc[:, :-1]
    n_instances, n_timepoints = X_test.shape
    X_test = np.reshape(X_test.values, (n_instances, 1, n_timepoints))

    y_test = df_test.iloc[:, -1]

    # 计算类别权重（处理不平衡数据集）
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    # 初始化 MiniRocket 特征提取器
    minirocket = MiniRocket(num_kernels=num_k)

    # 适应训练数据并转换它
    minirocket.fit(X_train)
    X_train_transform = minirocket.transform(X_train)

    # 使用转换后的特征训练线性分类器
    # 这里我们使用 RidgeClassifierCV 并且加入了计算出的类别权重
    # classifier = LogisticRegression(class_weight=class_weight_dict)
    classifier = RidgeClassifierCV(class_weight=class_weight_dict)
    classifier.fit(X_train_transform, y_train)

    # 对测试数据进行相同的特征提取过程
    X_test_transform = minirocket.transform(X_test)

    # 使用训练好的分类器进行预测
    y_pred = classifier.predict(X_test_transform)
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    # 打印到txt
    sup1 = 0
    with open(txt1, 'a') as f:
        f.write(f'case_{i}:-------------------------------------\n')
        for key, value in report.items():
            if key.isdigit():
                f.write(f"Class {key}:\n")
                f.write(
                    f"Precision={value['precision']:.4f}, Recall={value['recall']:.4f}, F1-score={value['f1-score']:.4f}, Support={value['support']:.4f}\n")
                if key == '1':
                    sup1 = value['support']
                    rec1 = value['recall']
                    num_1 = num_1 + sup1 * rec1

        acc = accuracy_score(y_true=y_test, y_pred=y_pred)
        pre = precision_score(y_true=y_test, y_pred=y_pred, average='weighted')
        rec = recall_score(y_true=y_test, y_pred=y_pred, average='weighted')
        f1 = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')
        gm = (rec * pre) ** 0.5
        f.write(f"acc：   w-pre:   rec:      f1:       gm:        \n")
        f.write(f"{acc:.4f}, {pre:.4f}, {rec:.4f}, {f1:.4f}, {gm:.4f}\n\n")

    # 打印到txt
    with open(txtmat, 'a') as f:
        acc = accuracy_score(y_true=y_test, y_pred=y_pred)
        pre = precision_score(y_true=y_test, y_pred=y_pred, average='weighted')
        rec = recall_score(y_true=y_test, y_pred=y_pred, average='weighted')
        f1 = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')
        gm = (rec * pre) ** 0.5
        f.write(f"{acc:.4f}, {pre:.4f}, {rec:.4f}, {f1:.4f}, {gm:.4f};\n")

with open(txtmat, 'a') as f:
    f.write(f"{num_1}/12\n")

    #控制台输出
    # for key, value in report.items():
    #     if key.isdigit():
    #         print(f"Class {key}:")
    #         print(
    #             f"Precision={value['precision']:.4f}, Recall={value['recall']:.4f}, F1-score={value['f1-score']:.4f}, Support={value['support']:.4f}")
    #         # for metric, score in value.items():
    #         #     print(f"{metric.capitalize()}: {score:.4f}")
    # acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    # pre = precision_score(y_true=y_test, y_pred=y_pred, average='weighted')
    # rec = recall_score(y_true=y_test, y_pred=y_pred, average='weighted')
    # f1 = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')
    # gm = (rec * pre) ** 0.5
    # print(f"acc:{acc:.4f}  weighted avg:: pre:{pre:.4f}  rec:{rec:.4f}  f1:{f1:.4f}  gm:{gm:.4f}")
    # print(f'case_{i}\n\n')

