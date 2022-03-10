#coding:gbk
import matplotlib.pyplot as plt
import random
import math

import pandas as pd

global m, C;  # m����Ʒ ,��������C
global time, balance;  # time ��������, balance  ƽ�����
global best, T, af;  # best �o��  T �¶�  af�˻���
m = 15;                    #��Ʒ��
T = 200.0;                 #�ض�
af = 0.95;                 #�˻���
time = 500;                #�����Δ�
balance = 500;             #ƽ��Δ�
best_way = [0] * m;        #best_way �o���ⷽ��
now_way = [0] * m          #now_way �o䛮�ǰ�ⷽ��
current_solution = []      #�Á��L����ǰ�⣬�L�D��
Best_solution = []         #�Á��L����ѽ⣬�L�D��
Number_of_iterations = []  #�Á��L�������Δ����L�D��

def cop(a, b, le):  # �}�u���� ��b�����ֵ�xֵ�oa���M
    for i in range(le):
        a[i] = b[i]


def calc(x):  # Ӌ�㱳���rֵ
    global C, wsum;
    vsum = 0;
    wsum = 0;
    for i in range(m):
        vsum += x[i] * value[i];
        wsum += x[i] * weight[i];
    return vsum;


def produce():  # ��ʼ�a����C��
    while (1 > 0):
        for k in range(m):                  # ��ʼ�a����C��
            if (random.random() < 0.5):
                now_way[k] = 1;
            else:
                now_way[k] = 0;
        calc(now_way)                       # �z�����ɵĽ��Ƿ�Ϸ�
        if (wsum < C): break;
    global best;
    best = calc(now_way);
    cop(best_way, now_way, m);              #���@�M��Ҳ�}�u�o��ѽ�


def init():  # ��ʼ������
    global C, T;
    C = max_size;
    produce()  # �a����ʼ��


def get(x):  # ��C���������Ѿ����ڵ���Ʒȡ��
    while (1 > 0):
        ob = random.randint(0, m - 1);
        if (x[ob] == 1): x[ob] = 0;break;


def put(x):  # ��C���뱳���в����ڵ���Ʒ
    while (1 > 0):
        ob = random.randint(0, m - 1);
        if (x[ob] == 0): x[ob] = 1;break;


def TXT_to_array(file_address):       #���Y���D���������
    f = open(file_address)            #�_�n
    val = []                          #���ִ�
    line = 1                          #��M��whileޒȦ��ʼ�l�� line�O1
    while line:                       #�x�n���x�����ֹ
        line = f.readline()           #����һ�Йn����ֵ�x��line

        val.append((line))            #��line�����ִ�val��

    val = val[:-1]                    #val����һ��Ԫ��''��������@����������
    val1 = list(map(int, val))        #��val�D�������ִ�val1
    return val1                       #����val1

value = TXT_to_array("C:/Users/craig/Desktop/01knapsack/p07_p.txt")  # ��Ʒ�rֵ
weight = TXT_to_array("C:/Users/craig/Desktop/01knapsack/p07_w.txt")  # ��Ʒ����
ww = TXT_to_array("C:/Users/craig/Desktop/01knapsack/p07_c.txt")  # ���ؓ��
max_size = ww[0]  # ���ؓ��


def slove():  # ��������
    global best, T, balance;
    test = [0] * m;
    now = 0;  # ��ǰ�����rֵ
    for i in range(balance):
        now = calc(now_way);
        cop(test, now_way, m);
        ob = random.randint(0, m - 1);  # �S�C�xȡĳ����Ʒ
        if (test[ob] == 1):
            put(test);test[ob] = 0;  # �ڱ����Єt�����ó����K����������Ʒ
        else:  # ���ڱ����Єtֱ�Ӽ������Q�����ڱ����е���Ʒ
            if (random.random() < 0.5):
                test[ob] = 1;
            else:
                get(test); test[ob] = 1;
        temp = calc(test);

        if (wsum > C): continue;  # �Ƿ���t���^
        if (temp > best): best = temp; cop(best_way, test, m);  # ����ȫ���

        if (temp > now):
            cop(now_way, test, m);  # ֱ�ӽ����½�
        else:
            g = 1.0 * (temp - now) / T;
            if (random.random() < math.exp(g)):  # ���ʽ����ӽ�
                cop(now_way, test, m);

    return now

            # *****************************������**********************


init();

print("��ʼ���:",now_way)

for i in range(time):                           #�_ʼ����
    solution = slove();                         #�a����ǰ��
    T = T * af;                                 #�¶��½�
    Number_of_iterations.append(i+1)            #�L�������Δ����L�D��
    current_solution.append(solution)           #�L����ǰ��
    Best_solution.append(max(current_solution)) #��ѽ⼯�ϼ��뮔ǰ�⼯�������ֵ

print('���:', best, '��������', time);
print('ȡ���飺', best_way);
print("Number_of_iterations:",Number_of_iterations)
print("Best_solution:",Best_solution)

plt.plot(Number_of_iterations, Best_solution, color='b')  #�����L�D����
plt.xlabel('Number of iterations') # �O��x�S���}
plt.ylabel('Best solution') # �O��y�S���}
plt.title('Simulated annealing') # �O���D����}
plt.show() #ݔ���D��