import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("C:/Users/craig/Desktop/train.csv")  #數據集資料引入
data.head()                                             #檢查數據集
data.columns                                            #檢視數據集的欄位
data.shape                                              #檢視數據集欄位數目和資料總數
data.dtypes                                             #檢視數據集的欄位變數
pd.isnull(data).sum()                                   #檢視是否含有null的資料
data.describe()                                         #檢視數據集的詳細情況
y = data['price_range']                                 #定義目標列price_range為y
x = data.drop('price_range', axis = 1)                  #將數據集的price_range欄移除並定義為x
y.unique()                                              #檢視目標列含有幾種不同的price數值
labels = ["low cost", "medium cost", "high cost", "very high cost"]   #將price分成4種
values = data['price_range'].value_counts().values                    #比率
colors = ['yellow','turquoise','lightblue', 'pink']                   #調色
fig1, ax1 = plt.subplots()
ax1.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)   #製成圓餅圖並填入參數
ax1.set_title('balanced or imbalanced?')                #title
plt.show()                                              #27-33行對4種price製圖並進行分析



x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2, random_state = 101, stratify = y)     #將數據集切分成訓練集'x_train y_train'與驗證集'x_valid y_valid'
print(x_train.shape)                                    #檢查訓練集x_train
print(x_valid.shape)                                    #檢查驗證集x_valid



fig = plt.subplots (figsize = (12, 12))
sns.heatmap(data.corr (), square = True, cbar = True, annot = True, cmap="GnBu", annot_kws = {'size': 8})
plt.title('Correlations between Attributes')
plt.show()                                              #43-46行製圖並分析屬性間的關係


#lr = LogisticRegression(multi_class = 'multinomial', solver = 'sag',  max_iter = 10000)     #lr module
#lr.fit(x_train, y_train)
#y_pred_lr = lr.predict(x_valid)
#confusion_matrix = metrics.confusion_matrix(y_valid, y_pred_lr)
#confusion_matrix
#acc_lr = metrics.accuracy_score(y_valid, y_pred_lr)
#acc_lr


svc=SVC(kernel='linear')                                   #svm module - linear kernel
svc.fit(x_train,y_train)                                   #丟入訓練集進行訓練
y_pred=svc.predict(x_valid)                                #使用訓練結果對驗證集進行預測
print('SVM linear kernel Accuracy Score:')
print(metrics.accuracy_score(y_valid,y_pred))              #查看linear kernel準確率
print(metrics.confusion_matrix(y_valid, y_pred))           #印出混淆矩陣
print(metrics.classification_report(y_valid, y_pred))      #印出評估表

svc=SVC(kernel='rbf')                                      #svm module - rbf kernel
svc.fit(x_train,y_train)                                   #丟入訓練集進行訓練
y_pred=svc.predict(x_valid)                                #使用訓練結果對驗證集進行預測
print('SVM rbf kernel Accuracy Score:')
print(metrics.accuracy_score(y_valid,y_pred))              #查看linear kernel準確率

svc=SVC(kernel='poly')                                     #svm module - poly kernel
svc.fit(x_train,y_train)                                   #丟入訓練集進行訓練
y_pred=svc.predict(x_valid)                                #使用訓練結果對驗證集進行預測
print('SVM poly kernel Accuracy Score:')
print(metrics.accuracy_score(y_valid,y_pred))              #查看linear kernel準確率




dt = DecisionTreeClassifier(random_state=101)              #decesion tree module
dt_model = dt.fit(x_train, y_train)                        #丟入訓練集進行訓練
y_pred_dt = dt.predict(x_valid)                            #使用訓練結果對驗證集進行預測
print(metrics.confusion_matrix(y_valid, y_pred_dt))        #印出混淆矩陣
print(metrics.classification_report(y_valid, y_pred_dt))   #印出評估表
acc_dt = metrics.accuracy_score(y_valid, y_pred_dt)
print('Decesion tree Accuracy Score:')                     #印出準確率
print(acc_dt)



rf = RandomForestClassifier(n_estimators = 100, random_state=101, criterion = 'entropy', oob_score = True)    #randomforest module
model_rf = rf.fit(x_train, y_train)                                                                           #丟入訓練集進行訓練
y_pred_rf = rf.predict(x_valid)                                                                               #使用訓練結果對驗證集進行預測
print(metrics.confusion_matrix(y_valid, y_pred_rf))                                                           #印出混淆矩陣
print(metrics.classification_report(y_valid, y_pred_rf))                                                      #印出評估表
pd.crosstab(y_valid, y_pred_rf, rownames=['Actual Class'], colnames=['Predicted Class'])
acc_rf = metrics.accuracy_score(y_valid, y_pred_rf)
print(' RandomForest Accuracy Score:')                                                                        #印出準確率
print(acc_rf)


model_knn = KNeighborsClassifier(n_neighbors=3)           #KNeighbors  module n_neighbors=3
model_knn.fit(x_train, y_train)                           #丟入訓練集進行訓練
y_pred_knn = model_knn.predict(x_valid)                   #使用訓練結果對驗證集進行預測
print(accuracy_score(y_valid, y_pred_knn))
parameters = {'n_neighbors':np.arange(1,30)}
knn = KNeighborsClassifier()
model = GridSearchCV(knn, parameters, cv=5)
model.fit(x_train, y_train)
model.best_params_

model_knn = KNeighborsClassifier(n_neighbors=9)           #KNeighbors  module n_neighbors=9
model_knn.fit(x_train, y_train)                           #丟入訓練集進行訓練
y_pred_knn = model_knn.predict(x_valid)                   #使用訓練結果對驗證集進行預測
print(metrics.confusion_matrix(y_valid, y_pred_knn))      #印出混淆矩陣
print(metrics.classification_report(y_valid, y_pred_knn)) #印出評估表
acc_knn = accuracy_score(y_valid, y_pred_knn)
print('KNeighbors Accuracy Score:')                       #印出準確率
print(acc_knn)



models = ['decision tree', 'random forest', 'knn', 'svm']
acc_scores = [0.83, 0.90, 0.95, 0.98]                                     #將4種模型的accuracy當作比較對象
plt.bar(models, acc_scores, color=['lightblue', 'pink', 'lightgrey', 'cyan'])
plt.ylabel("accuracy scores")
plt.title("Which model is the most accurate?")
plt.show()                                                                #對4種模型accuracy製圖並比較


test_data = pd.read_csv("C:/Users/craig/Desktop/test.csv")                #匯入測試集
test_data.head()                                                          #檢視測試集
test_data=test_data.drop('id',axis=1)                                     #刪除測試集的'id'欄位
test_data.head()
predicted_price_range = svc.predict(test_data)                            #使用準確率較高的svm模型預測price
test_data['price_range'] = predicted_price_range                          #測試集新增price_range欄位並填入預測值

print(test_data)                                                          #輸出測試集test_data
print(x_train)                                                            #輸出訓練集x_train
print(y_train)                                                            #輸出訓練集y_train
print(x_valid)                                                            #輸出驗證集x_valid
print(y_valid)                                                            #輸出驗證集y_valid



