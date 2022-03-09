def knapsack(W, wt, val, n):                                   #0/1 knapsack數學定義式寫出的函數
    dp = [[0 for i in range(W + 1)] for j in range(n + 1)]     #構建dp圖表:i = 0 to w ,j = 0 to n
    for i in range(1, n + 1):                                  #for i = 1 to n do
        for j in range(1, W + 1):                              #for j = 1 to W do
            if wt[i - 1] <= j:                                 #拿得動物品i
                dp[i][j] = max(val[i - 1] + dp[i - 1][j - wt[i - 1]], dp[i - 1][j])  #比較拿或不拿哪個獲益比較高
            else:                                              #拿不動物品i
                dp[i][j] = dp[i - 1][j]                        #則開始考慮下個物品
    return dp[n][W], dp                                        #傳回最佳解dp[n][W]和dp表


def knapsack_with_example_solution(W: int, wt: list, val: list):   #主要的函數
    num_items = len(wt)                                            #求出物品數，並傳給_construct_solution函數
    optimal_val, dp_table = knapsack(W, wt, val, num_items)    #由knapsack函數求出最佳解和dp表
    example_optional_set: set = set()
    _construct_solution(dp_table, wt, num_items, W, example_optional_set)  #由_construct_solution函數求出物品取法
    return optimal_val, example_optional_set                               #傳回最佳解和取法


def _construct_solution(dp: list, wt: list, i: int, j: int, optimal_set: set):  #求出最後取出的物品編號
    if i > 0 and j > 0:
        if dp[i - 1][j] == dp[i][j]:
            _construct_solution(dp, wt, i - 1, j, optimal_set)
        else:
            optimal_set.add(i)
            _construct_solution(dp, wt, i - 1, j - wt[i - 1], optimal_set)

def TXT_to_array(file_address):       #將資料轉成整數矩陣
    f = open(file_address)            #開檔
    val = []                          #空字串
    line = 1                          #為滿足while迴圈初始條件 line設1
    while line:                       #讀檔案讀到完為止
        line = f.readline()           #將下一行檔案的值賦予line

        val.append((line))            #將line加入字串val中

    val = val[:-1]                    #不知道為甚麼寫出來val會多一個元素''，於是用這個把它砍掉
    val1 = list(map(int, val))        #把val轉成整數字串val1
    return val1                       #傳回val1



if __name__ == "__main__":
    val = TXT_to_array("C:/Users/craig/Desktop/01/p07_p.txt")   #用TXT_to_array取得val陣列
    wt = TXT_to_array("C:/Users/craig/Desktop/01/p07_w.txt")    #用TXT_to_array取得wt陣列
    n = len(val)                                                #取得物品數n
    ww = TXT_to_array("C:/Users/craig/Desktop/01/p07_c.txt")    #用TXT_to_array取得ww陣列
    w = ww[0]                                                   #把最大負重w從ww陣列中取出
    F = [[0] * (w + 1)] + [[0] + [-1 for i in range(w + 1)] for j in range(n + 1)]
    optimal_solution, optimal_subset = knapsack_with_example_solution(w, wt, val)  #求出最佳解和取法
    print("optimal_value = ", optimal_solution)                 #印出最佳解
    print("An optimal subset corresponding to the optimal value", optimal_subset)  #印出取法