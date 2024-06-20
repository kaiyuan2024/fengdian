import numpy as np
import pandas as pd
import os
caseName = ['6F2']
arguments = ['BusVolMag', 'BusVolAng', 'LinCurMag', 'LinCurAng', 'RotorAng', 'RotorSpd']
for i_case in range(0,1):
    txt_filepath = r'D:\python_work\tsai-main-v39\main\vali1706\data179\sourceTxt\Case '+f'{caseName[i_case]}'+'\\'
    for i_arg in range(0,len(arguments)):
        path_now = txt_filepath + arguments[i_arg] + '.txt'
        print("Reading ",path_now,"……")

        with open(path_now, 'r') as file:
            names = file.readline().split("' '")
            names[-1] = names[-1].replace("' \n", '')   #去除末尾的回车和单引号。单引号去掉是可以保持和其他name的格式一致，time不去是因为和数据列名字本来也不一个格式
            print("lenth of columns: ", len(names), '\n')

        if len(set(names)) != len(names):
            print("存在重复的列名:")
            seen = set(names)
            for inx, name in enumerate(names):
                if name in seen:
                    seen.remove(name)
                else:
                    names[inx] = name+'_duplicate' #这个方法只能解决两个重复列名
                    print("列号:",inx," 原来的列名:",name," 改名后的列名:",names[inx], '\n')
        else:
            print("没有重复的列名。",'\n')


        df = pd.read_csv(path_now, header=0, names=names, sep="\s{1,4}", engine="python", encoding="utf-8")
        print("读到的dataFrame大小:",df.shape, '\n')
        print(df.head(3), '\n')
        print(df.describe(), '\n')

        #经过处理依据能读取数据到df里，下面有两种保存：
        #创建文件夹
        save_filepath = r"D:\python_work\tsai-main-v39\main\vali1706\data179\csvData"
        os.makedirs(save_filepath, exist_ok=True)   #路径已存在时，忽略创建命令
        #1存为csv，一格一格，不用四处设置分隔符，应该能更快地读取处理了
        # csv_name = r"D:\python_work\ts-ai-main(open-source deep learning package built for time series)\data\WECC240\case"+f'{i_case}'+"\\"+f'{arguments[i_arg]}'+".csv"
        csv_name = save_filepath+'\\case'+f'{caseName[0]}'+f'{arguments[i_arg]}'+".csv"
        df.to_csv(csv_name, mode='w')   #覆写
        # dfload = pd.read_csv(csv_name, index_col=0)

    print('Case'+f'{caseName[i_case]}'+' Done!!!')



##pandas观察dataFrame的方法-----------------------------------------------------------------------------------------------
# print(BusVolMag.shape)      #输出dataframe 有多少行多少列
# print(BusVolMag.shape[0])   #取行数
# print(BusVolMag.shape[1])     #取列数
# print(BusVolMag.columns)          #顺序输出每一列的名字，是一个列表
# print(BusVolMag.index)       #顺序输出每一行的名字，
# print(BusVolMag.dtypes)      # 数据每一列的类型不一样，比如数字、字符串、日期等。该方法输出每一列变量类型
# print(BusVolMag.head(3))          #看前3行的数据，默认为5
# print(BusVolMag.tail(3))             #看最后3行的数据，默认为5
# print(BusVolMag.sample(frac=0.5))       #随机抽取3行，想要的固定比例的话，可以用frac参数
# print(BusVolMag.describe())           #非常方便的函数，对每一列数据有一个直管感受；只会对数字类型的列有效
# BusVolMag.get("salary")  # 取某一列

##用pandas检查是否有重复列-------------------------------------------------------------------------------------------------
# import pandas as pd

# 创建包含重复列名的示例数据帧——转换数据类型
# # data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'A': [7, 8, 9]}
# df = pd.DataFrame(title_BusVolMag)
#
# # 检查是否存在重复的列名
# duplicate_rows = df.duplicated(keep=False)
# if sum(duplicate_rows) > 0:
#     print("存在重复的列名:", [i for i in duplicate_rows.index if duplicate_rows[i]]) #也可以写作duplicate_rows[duplicate_rows].index，输出形式不太一样
# else:
#     print("没有重复的列名")

##用pandas检查是否有重复列-再删除重复列--------------------------------------------------------------------------------------
    # duplicate_rows = df.T.duplicated(keep=False)
    # if sum(duplicate_rows) > 0:
    #     print("存在重复的列名:", [i for i in duplicate_rows.index if duplicate_rows[i]])  # 也可以写作duplicate_rows[duplicate_rows].index，输出形式不太一样
    #     df = df.drop_duplicates()
    #     print("已删除重复列名，数据shape：",df.shape, '\n\n')
    # else:
    #     print("没有重复的列名。")

##打印列表---------------------------------------------------------------------------------------------------------------
# for i in range(0,len(title_BusVolMag)):
#     print(title_BusVolMag[i])







