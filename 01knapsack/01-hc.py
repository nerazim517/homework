import random
import matplotlib.pyplot as plt

time = 500  #跌代次數
best_solution = {}
initial_solution = {}
profit_array = []
Number_of_iterations = []

def TXT_to_array(file_address):       #將資料轉成整數矩陣
    f = open(file_address)            #開檔
    val = []                          #空字串
    line = 1                          #為滿足while迴圈初始條件 line設1
    while line:                       #讀檔案讀到完為止
        line = f.readline()           #將下一行檔案的值賦予line
        val.append((line))            #將line加入字串val中
    val = val[:-1]                    #寫出來val會多一個元素''，於是用這個把它砍掉
    val1 = list(map(int, val))        #把val轉成整數字串val1
    return val1                       #傳回val1


def count_weight(list , weight):     #計算負重
    sum = 0
    for i in range(len(list)):
        if(list[i] == 1):
            sum = sum + weight[i]
        i = i+1
    return sum

def count_value(list , profit1):     #計算獲利
    sum = 0
    for i in range(len(list)):
        if(list[i] == 1):
            sum = sum + profit1[i]
        i = i+1
    return sum


def random_int_list(start, stop, length):   #產生隨機01字串
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list


def neighbor(list1):                          #產生鄰近解
    list = list1[:]                           #賦值並讓兩變數互相獨立
    x = random.randint(0, len(list)-1)        #隨機改變1位數字
    if(list[x] == 0):                         #0(不取)變1(取)
        list[x] = 1
    else:
        list[x] = 0                           #1(取)變0(不取)

    return list                               #傳回差1位數的新解

def iteration(iter,list,profit,weight,capacity):                    #跌代

    best_solution_now = list[:]               #賦值並讓兩變數互相獨立
    for i in range(iter):                     #開始跌代
        Number_of_iterations.append(i+1)      #畫圖用的變數，圖的x軸
        new = neighbor(best_solution_now)     #產生一組差1位數的新解(鄰居)
        if((count_value(best_solution_now , profit) < count_value(new , profit))&(count_weight(new , weight)<=capacity)):
            best_solution_now = new           #如果鄰居的值比較大而且沒有超過背包付重上限的話，鄰居的解就是目前最佳解
        profit_array.append(count_value(best_solution_now , profit))   #畫圖用的變數，紀錄每次跌代後最佳解的數值
    return best_solution_now                  #跌代完成後傳回最佳解




#----------------------------------------------------------------------------------------------------
weight = TXT_to_array("C:/Users/craig/p07_w.txt")  #物重，共15物
profit = TXT_to_array("C:/Users/craig/p07_p.txt")  #獲利
capacity = TXT_to_array("C:/Users/craig/p07_c.txt")  #容量



while True:                                 #產生合法初始解
    solution = random_int_list(0,1,15)
    initial_solution = solution
    best_solution = solution
    if(count_weight(initial_solution,weight)<=capacity[0]):
        break

print("initial solution:",initial_solution)
print("initial profit:",count_value(initial_solution,profit))
best_solution = iteration(time,initial_solution,profit,weight,capacity[0])
print("best solution:",best_solution)
print("best profit:",count_value(best_solution,profit))
print(profit_array)         #記錄每次跌代後最佳獲利
plt.plot(Number_of_iterations, profit_array, color='b')  #帶入繪圖參數
plt.xlabel('Number of iterations') # 設定x軸標題
plt.ylabel('Best solution') # 設定y軸標題
plt.title('Hill climbing') # 設定圖表標題
plt.show() #輸出圖表