import pandas as pd

# 假设 df 是你的 DataFrame
# 这里创建一个示例 DataFrame，2700 个特征列和一个标签列
# df = pd.DataFrame(你的数据)

# 设置窗口大小和步长
window_size = 2100
step_size = 60
feature_columns = 2700

for i in range(1, 14):
    df = pd.read_csv(f'WECC240_U/case{i}BusVolMag.csv', index_col=0)
# 初始化一个空的 DataFrame，用于存放新的多行样本
    new_rows = []

    # 遍历每一行
    for index, row in df.iterrows():

        # 提取特征列和标签列
        features = row[:feature_columns]
        label = row[feature_columns]

        # 滑动窗口处理特征列
        for start in range(0, feature_columns - window_size + 1, step_size):
            end = start + window_size
            window = features[start:end]
            new_row = window.tolist() + [label]  # 将窗口特征和标签合并为新行
            new_rows.append(new_row) # 添加新行到DataFrame

    # 重置索引，因为我们连续添加了多行
    new_df = pd.DataFrame(new_rows)

    # 如果需要，可以给新的 DataFrame 添加列名
    # new_df.columns = ['Feature1', 'Feature2', ..., 'Label']

    print(new_df.shape)
    new_df.to_csv(f'w2100s60/case{i}BusVolMag.csv')
