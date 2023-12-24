from sklearn.datasets import make_blobs #產生數據集
from sklearn.preprocessing import StandardScaler #標準化
from sklearn.model_selection import train_test_split #分割訓練資料
from sklearn.linear_model import LogisticRegression #邏輯斯
import matplotlib.pyplot as plt


#產生數據集
dx, dy = make_blobs(n_samples=500, n_features=2, centers=2, random_state=0)
dx_std = StandardScaler().fit_transform(dx)


#分割訓練資料
dx_train, dx_test, dy_train, dy_test = train_test_split(dx_std,dy,test_size=0.2,random_state=0)


#建立邏輯斯模型
log_reg = LogisticRegression()
#輸入訓練集資料
log_reg.fit(dx_train,dy_train)
#使用測試集資料做預測
predictions = log_reg.predict(dx_test)


#印出實際的測試集標籤資料
print(dy_test)
#印出預測的標籤資料
print(predictions)


#模型對訓練集的預測準確率
print(log_reg.score(dx_train,dy_train))
print(log_reg.score(dx_test,dy_test))
