from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt


# 取得標籤資料，並分為訓練集與測試集
dx, dy = load_breast_cancer(return_X_y=True)
dx_train, dx_test, dy_train, dy_test = train_test_split(dx, dy,test_size=0.2,random_state=0)


# 用來收集交叉驗證準確率的list
cv_scores = []
# 用來收集測試準確率的list
test_scores = []
# 圖表x軸，也就是k值，1-10
x = np.arange(10) + 1


# 利用for迴圈，走訪完arange裡list所有內容
# KNeighborsClassifier()為建立模型
# fit()為訓練模型
for k in x:
    knn = KNeighborsClassifier(n_neighbors=k).fit(dx_train, dy_train)
    cv_scores.append(cross_val_score(knn, dx_train, dy_train, cv=5).mean())
    test_scores.append(knn.score(dx_test,dy_test))

plt.title('KNN hyperparameter')
plt.plot(x, cv_scores, label='CV score')
plt.plot(x, test_scores, label='Test score')
plt.xlabel('k neighbors')
plt.ylabel('accurancy(%)')
plt.legend() # 建立圖例
plt.grid(True)
plt.show()