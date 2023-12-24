from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


#從葡萄酒資料集讀取資料，將特徵與標籤資料一起傳回
dx,dy = load_wine(return_X_y=True)


#資料標準化
dx_std = StandardScaler().fit_transform(dx)


#分割訓練資料
dx_train, dx_test, dy_train, dy_test = train_test_split(dx_std,dy,test_size=0.2,random_state=0)


#建立決策樹模型
forest = RandomForestClassifier()
#輸入訓練集資料
forest.fit(dx_train,dy_train)
#進行k=5交叉驗證
val_score = cross_val_score(forest, dx_train, dy_train, cv=5)
#使用測試集資料做預測
predictions = forest.predict(dx_test)


#模型對訓練集的準確率(小數點第三位)
print(forest.score(dx_train,dy_train).round(3))
#k-fold交叉驗證的準確率(小數點第三位)
print(val_score.mean().round(3))
#模型對測試集的準確率(小數點第三位)
print(forest.score(dx_test,dy_test).round(3))