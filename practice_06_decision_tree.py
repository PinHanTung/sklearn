from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


#從鳶尾花資料集讀取資料，將特徵與標籤資料一起傳回
dx,dy = load_iris(return_X_y=True)


#分割訓練資料
dx_train, dx_test, dy_train, dy_test = train_test_split(dx,dy,test_size=0.2,random_state=0)


#建立決策樹模型
tree = DecisionTreeClassifier()
#輸入訓練集資料
tree.fit(dx_train,dy_train)
#使用測試集資料做預測
predictions = tree.predict(dx_test)


#模型對訓練集的預測準確率
print(tree.score(dx_train,dy_train))
print(tree.score(dx_test,dy_test))