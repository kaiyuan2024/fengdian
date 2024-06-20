import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import ADASYN
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
# 假设 X_train, y_train 分别是你的训练数据特征和标签


txt1 = "ad_lr2.txt"
txtmat = "ad_lr2-mat.txt"
num_1 = 0
l_df = []
for i in range(1, 14):
    df = pd.read_csv(f'WECC240_U/case{i}BusVolMag.csv', index_col=0)
    df.dropna(inplace=True)

    l_df.append(df)
for i in range(1, 14):
    df_test = l_df[i - 1]
    df_train = pd.concat(l_df[:i - 1] + l_df[i:], axis=0)

    # (n_instances, n_variables, n_timepoints)
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]

    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

    # 查看经过ADASYN处理后的类分布
    print(f"Class distribution after ADASYN: {np.bincount(y_resampled)}")

    # 使用随机森林分类器
    rf_clf = LogisticRegression()
    rf_clf.fit(X_resampled, y_resampled)

    # 使用训练好的分类器进行预测
    y_score = rf_clf.predict_proba(X_test)[:, 1]
    print(y_score[:10])
    y_pred = np.where(y_score > 0.4, 1, 0)
    print(f'test case {i}')
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
