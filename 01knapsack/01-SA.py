#coding:gbk
import matplotlib.pyplot as plt
import random
import math

import pandas as pd

global m, C;  # m個物品 ,背包容量C
global time, balance;  # time 迭代次数, balance  平衡次数
global best, T, af;  # best 紀錄最優  T 温度  af退火率
m = 15;                    #物品數
T = 200.0;                 #溫度
af = 0.95;                 #退火率
time = 500;                #迭代次數
balance = 500;             #平衡次數
best_way = [0] * m;        #best_way 紀錄最優解方案
now_way = [0] * m          #now_way 紀錄當前解方案
current_solution = []      #用來蒐集當前解，繪圖用
Best_solution = []         #用來蒐集最佳解，繪圖用
Number_of_iterations = []  #用來蒐集迭代次數，繪圖用

def cop(a, b, le):  # 複製函數 把b數组的值賦值給a數組
    for i in range(le):
        a[i] = b[i]


def calc(x):  # 計算背包價值
    global C, wsum;
    vsum = 0;
    wsum = 0;
    for i in range(m):
        vsum += x[i] * value[i];
        wsum += x[i] * weight[i];
    return vsum;


def produce():  # 初始產生随機解
    while (1 > 0):
        for k in range(m):                  # 初始產生随機解
            if (random.random() < 0.5):
                now_way[k] = 1;
            else:
                now_way[k] = 0;
        calc(now_way)                       # 檢查生成的解是否合法
        if (wsum < C): break;
    global best;
    best = calc(now_way);
    cop(best_way, now_way, m);              #將這組解也複製給最佳解


def init():  # 初始化函数
    global C, T;
    C = max_size;
    produce()  # 產生初始解


def get(x):  # 随機將背包中已经存在的物品取出
    while (1 > 0):
        ob = random.randint(0, m - 1);
        if (x[ob] == 1): x[ob] = 0;break;


def put(x):  # 随機放入背包中不存在的物品
    while (1 > 0):
        ob = random.randint(0, m - 1);
        if (x[ob] == 0): x[ob] = 1;break;


def TXT_to_array(file_address):       #將資料轉成整數矩陣
    f = open(file_address)            #開檔
    val = []                          #空字串
    line = 1                          #為滿足while迴圈初始條件 line設1
    while line:                       #讀檔案讀到完為止
        line = f.readline()           #將下一行檔案的值賦予line

        val.append((line))            #將line加入字串val中

    val = val[:-1]                    #val會多一個元素''，於是用這個把它砍掉
    val1 = list(map(int, val))        #把val轉成整數字串val1
    return val1                       #傳回val1

value = TXT_to_array("C:/Users/craig/Desktop/01knapsack/p07_p.txt")  # 物品價值
weight = TXT_to_array("C:/Users/craig/Desktop/01knapsack/p07_w.txt")  # 物品重量
ww = TXT_to_array("C:/Users/craig/Desktop/01knapsack/p07_c.txt")  # 最大負重
max_size = ww[0]  # 最大負重


def slove():  # 迭代函数
    global best, T, balance;
    test = [0] * m;
    now = 0;  # 當前背包價值
    for i in range(balance):
        now = calc(now_way);
        cop(test, now_way, m);
        ob = random.randint(0, m - 1);  # 隨機選取某個物品
        if (test[ob] == 1):
            put(test);test[ob] = 0;  # 在背包中則將其拿出，並加入其它物品
        else:  # 不在背包中則直接加入或替換掉已在背包中的物品
            if (random.random() < 0.5):
                test[ob] = 1;
            else:
                get(test); test[ob] = 1;
        temp = calc(test);

        if (wsum > C): continue;  # 非法解則跳過
        if (temp > best): best = temp; cop(best_way, test, m);  # 更新全局最優

        if (temp > now):
            cop(now_way, test, m);  # 直接接受新解
        else:
            g = 1.0 * (temp - now) / T;
            if (random.random() < math.exp(g)):  # 概率接受劣解
                cop(now_way, test, m);

    return now

            # *****************************主函数**********************


init();

print("初始解為:",now_way)

for i in range(time):                           #開始迭代
    solution = slove();                         #產生當前解
    T = T * af;                                 #温度下降
    Number_of_iterations.append(i+1)            #蒐集迭代次數，繪圖用
    current_solution.append(solution)           #蒐集當前解
    Best_solution.append(max(current_solution)) #最佳解集合加入當前解集合中最大值

print('最優解:', best, '迭代次数', time);
print('取法為：', best_way);
print("Number_of_iterations:",Number_of_iterations)
print("Best_solution:",Best_solution)

plt.plot(Number_of_iterations, Best_solution, color='b')  #帶入繪圖參數
plt.xlabel('Number of iterations') # 設定x軸標題
plt.ylabel('Best solution') # 設定y軸標題
plt.title('Simulated annealing') # 設定圖表標題
plt.show() #輸出圖表