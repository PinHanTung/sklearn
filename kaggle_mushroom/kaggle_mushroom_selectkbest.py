import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_columns', None)
pd.options.mode.use_inf_as_na = True


# 從本地匯入資料、注意反斜
path = r"C:\Users\a0920\Documents\GitHub\sklearn\kaggle_mushroom\mushrooms.csv"
d1 = pd.read_csv(path,sep=',')


# 打印頭幾行、行列數量、確認是否有NA與空值
#print(d1.head())
#print(d1.shape)
#print(d1.info())
#print(pd.isnull(d1).sum())


# 創建一個LabelEncoder對象
label_encoder = LabelEncoder()
d1 = d1.apply(label_encoder.fit_transform)


# 將資料的class獨立出來成為標籤資料dy，其餘為特徵資料dx
dy = d1.iloc[:,0]
dx = d1.iloc[:,1:]
#print(dx[['ring-type','gill-color','gill-size']].head())
#print(dy.head())


'''
# 計算卡方統計量和 p 值
chi2_stat, p_values = chi2(dx, dy)


# 查看 p 值從小到大的排序，與其對應特徵
col = pd.DataFrame(dx.columns, columns=["Feature"])
col_p = pd.concat([col,pd.DataFrame(p_values, columns=["P-value"])],axis=1)
print(col_p.sort_values(['P-value']))
'''


# 將前三特徵篩選出來，再進行knn，看有何差別
dx = SelectKBest(chi2, k=3).fit_transform(dx, dy)
print(pd.DataFrame(dx).head())

# 分割資料集
dx_train, dx_test, dy_train, dy_test = train_test_split(dx,dy,test_size=0.2,random_state=0)
'''
print('dx.shape:',dx.shape) #(8124, 22)
print('分割後的 dx_train.shape:',dx_train.shape) 
print('分割後的 dx_test.shape:',dx_test.shape) 
'''


# 建立KNN模型、用GridSearchCV來尋找最佳參數並套入
param_grid = {'n_neighbors':np.arange(10)+1}
mushroom = GridSearchCV(KNeighborsClassifier(),param_grid)
'''
print('網格後的 dx_train:',dx_train.shape)
print('網格後的 dx_test:',dx_test.shape)
print('網格後的 dy_train:',dy_train.shape)
print('網格後的 dy_test:',dy_test.shape)
'''


# 訓練模型、k-fold交叉驗證
mushroom.fit(dx_train, dy_train)


val_score = cross_val_score(mushroom,dx_train, dy_train, cv=5)


# 預測分數
print('模型對訓練集的準確率:',mushroom.score(dx_train,dy_train).round(3))
print('k-fold交叉驗證的準確率:',val_score.mean().round(3))
print('模型對測試集的準確率:',mushroom.score(dx_test,dy_test).round(3))
print('---')
print('最佳參數:',mushroom.best_params_)
print('GridSearchCV交叉驗證的準確率:',mushroom.best_score_.round(3))