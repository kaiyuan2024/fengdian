import os
import re
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

txt_filepath_root = r"D:\python_work\tsai-main-v39\main\data\WECC240\case"
arguments = ['BusVolMag', 'BusVolAng', 'LinCurMag', 'LinCurAng']  #['LinCurAng']
for i_case in range(1,14):#14
    for i_arg in range(0,len(arguments)):
        #load data
        csv_name = txt_filepath_root + f'{i_case}' + "\\" + f'{arguments[i_arg]}' + ".csv"
        df_load = pd.read_csv(csv_name, index_col=0)

        # #截取2700行 删掉time列
        # df_cut = df_load.iloc[0:2700,:].drop(labels=["'Time"], axis=1)
        #截取后1200行 删掉time列 并保留time用作时间轴
        df_cut = df_load.iloc[0:2700, :]
        time_axis = df_cut.pop("'Time")

        # 数据观察？

        #转置后删除重复行
        df_T = df_cut.T
        duplicate_rows = df_T.duplicated(keep=False)
        if sum(duplicate_rows) > 0:
            print("存在重复的样本:", [i for i in duplicate_rows.index if duplicate_rows[i]])  # 也可以写作duplicate_rows[duplicate_rows].index，输出形式不太一样
            df_T.drop_duplicates(inplace=True)
            print("已删除重复样本，数据shape：",df_T.shape, '\n')

        # phase unwrapped
        if i_arg in [1, 3]:#
            # 观察uwrap前后的变化
            # plt.figure()
            # plt.plot(time_axis, df_T.T)
            # plt.show(block=False)
            rad = np.deg2rad(df_T)
            rad_unwrap = np.unwrap(rad, discont=np.deg2rad(150), axis=-1)   #numpy1.21版本该函数有变化
            df_T_unwrap = np.rad2deg(rad_unwrap)
            df_T[:] = df_T_unwrap
            #观察uwrap前后的变化
            # plt.figure()
            # plt.plot(time_axis, df_T_unwrap.T)
            # plt.show(block=False)

        # 存储变量，供后续计算功率等参数（#先去除最末回车符（否则不能赋值））
        exec(f"{arguments[i_arg]}_title = df_T.index")
        exec(f"{arguments[i_arg]} = df_T.values")

        #获取bus标签
        if i_case == 1:
            #df_T['target'] = np.where("1431|PALOVRD2    20.0" == df_T.index, 1, 0)
            #df_T['target'] = np.where(df_T.index.str.contains("1431"), 1, 0)
            target_np = np.where(df_T.index.str.contains("1431"), 1, 0)
        elif  i_case == 2:
            target_np = np.where(df_T.index.str.contains("2634"), 1, 0)
        elif  i_case == 3:
            target_np = np.where(df_T.index.str.contains("1131"), 1, 0)
        elif  i_case == 4:
            target_np = np.where(df_T.index.str.contains("3831"), 1, 0)
        elif  i_case == 5:
            target_np = np.where(df_T.index.str.contains("4231"), 1, 0)
        elif  i_case == 6:
            target_np = np.where(df_T.index.str.contains("7031"), 1, 0)
        elif i_case == 7:
            target_np = np.where(df_T.index.str.contains("2634"), 1, 0)
        elif i_case == 8:
            target_np = np.where(df_T.index.str.contains("6333"), 1, 0)
        elif i_case == 9:
            target_np = np.where(df_T.index.str.contains("6533|4131"), 1, 0)
        elif i_case == 10:
            target_np = np.where(df_T.index.str.contains("3931|6335"), 1, 0)
        elif i_case == 11:
            target_np = np.where(df_T.index.str.contains("4009"), 1, 0)
        elif i_case == 12:
            target_np = np.where(df_T.index.str.contains("6335"), 1, 0)
        elif i_case == 13:
            target_np = np.where(df_T.index.str.contains("4010|2619"), 1, 0)

        # 计算功率
        if i_arg == 3:
            pattern = r'\d+\.?\d*'
            power_P = df_T.copy()
            power_Q = df_T.copy()
            WD = pd.DataFrame(0, index=df_T.index, columns=range(0, 2700))

            #ab
            eps = 0.05
            fo = [0.82, 1.19, 0.379, 0.379, 0.68, 1.27, 0.379, 0.614, 0.762, 0.614, 0.614, 0.74, 0.614]###############
            wno = fo[i_case-1]/15
            N, Wn = signal.buttord([(1 - eps) * wno, (1 + eps) * wno], [(1 - 2*eps) * wno, (1 + 2*eps) * wno], gpass=5, gstop=15)
            b, a = signal.butter(N, Wn, 'bandpass')
            # w, h = signal.freqz(b,a)
            # plt.plot(w, 20*np.log10(abs(h)))

            #match_p = np.zeros((len(LinCurMag_title), 2))
            # count = 0   #寻找I的流入/流出端口号是否都能在U的端口号找到对应的。line1
            for i_i in range(0,len(LinCurMag_title)):
                match_i = re.findall(pattern, LinCurMag_title[i_i])
                # if match_i[4] == '2': print(LinCurMag_title[i_i],'********')   #寻找剔除重复数据后剩余的2号线
                for i_u in range(0,len(BusVolMag_title)):
                    match_u = re.findall(pattern, BusVolMag_title[i_u])
                    if match_u[0] == match_i[0]:   #电流流出节点match_i[0],电流流入节点match_i[2]
                        fai = BusVolAng[i_u] - LinCurAng[i_i]
                        # plt.figure()
                        # plt.plot(time_axis, fai)
                        # plt.show(block=False)
                        S = np.exp(1j * np.deg2rad(fai)) * BusVolMag[i_u] * LinCurMag[i_i]
                        power_P.iloc[i_i] = S.real   #np.real(S)
                        power_Q.iloc[i_i] = S.imag

                        d_Pi = signal.filtfilt(b, a, S.real)
                        d_Qi = signal.filtfilt(b, a, S.imag)
                        d_Ai = signal.filtfilt(b, a, BusVolAng[i_u])
                        d_Vi = signal.filtfilt(b, a, BusVolMag[i_u])
                        V_avr = np.mean(BusVolMag[i_u])
                        V_xing = d_Vi + V_avr*np.ones(2700)#np.expand_dims(V_avr,1).repeat(1200,axis=1)
                        for n_t in range(0, 2699):
                            # WD.iloc[i_i,n_t+1] = WD.iloc[i_i,n_t] + S.real[n_t]*(np.deg2rad(BusVolAng[i_u,n_t+1]-BusVolAng[i_u,n_t])) + S.imag[n_t]*(BusVolMag[i_u,n_t+1]-BusVolMag[i_u,n_t])/V_xing[i_u,n_t]
                            WD.iloc[i_i,n_t+1] = WD.iloc[i_i,n_t] + d_Pi[n_t]*(np.deg2rad(d_Ai[n_t+1]-d_Ai[n_t])) + d_Qi[n_t]*(d_Vi[n_t+1]-d_Vi[n_t])/V_xing[n_t]


                        #match_p[i_i, 0:2] = [i_i, i_u]   #0:2是第0列到第2列（不包括第2列）的切片
                        # count += 1   #寻找I的流入/流出端口号是否都能在U的端口号找到对应的。line2
            # if count != (i_i+1): print('not all---------------------')   #寻找I的流入/流出端口号是否都能在U的端口号找到对应的。line3
            # 画图观察
            # plt.figure()
            # plt.plot(time_axis, power_Q.T)
            # plt.show(block=False)

            # 添加target列
            # power_P['target'] = target_np
            # power_Q['target'] = target_np
            WD['target'] = target_np

            #save data
            save_filepath = r"D:\python_work\tsai-main-v39\main\data\WECC240_csv1"
            os.makedirs(save_filepath, exist_ok=True)  #创建文件夹   #路径已存在时，忽略创建命令
            # csv_name = save_filepath + r"\case" + f'{caseNamei}' + "PowerP" + ".csv"
            # power_P.to_csv(csv_name)  #power_P.T.iloc[900:1800, :].to_csv(csv_name)
            # csv_name = save_filepath + r"\case" + f'{caseNamei}' + "PowerQ" + ".csv"
            # power_Q.to_csv(csv_name)
            csv_name = save_filepath + r"\case" + f'{i_case}' + "WD" + ".csv"
            WD.to_csv(csv_name)

        # 添加target列
        df_T['target'] = target_np
        # 添加target行
        # df_T.loc[len(df_T.index)] = target_np
        # df_T = df_T.rename(index={df_T.index[-1]: 'target'})

        #save data
        # save_filepath = r"D:\python_work\tsai-main-v39\main\vali1706\data179\csvData3"
        # os.makedirs(save_filepath, exist_ok=True)  #创建文件夹   #路径已存在时，忽略创建命令
        # # arguments_csv = ['U', 'Uangle', 'I', 'Iangle']
        # csv_name = save_filepath + r"\case" + f'{caseNamei}' + f'{arguments[i_arg]}' + ".csv"
        # df_T.to_csv(csv_name)

    print('Case' + f'{i_case}' + ' Done.\n\n')
