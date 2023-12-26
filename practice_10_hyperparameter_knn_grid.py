from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt


# 取得標籤資料，並分為訓練集與測試集
dx, dy = load_breast_cancer(return_X_y=True)
dx_train, dx_test, dy_train, dy_test = train_test_split(dx, dy,test_size=0.2,random_state=0)


# 建立網格搜尋的參數(字典)、使用網格搜尋
param_grid = {'n_neighbors':np.arange(10)+1}
model = GridSearchCV(KNeighborsClassifier(),param_grid)


# 訓練模型
model.fit(dx_train, dy_train)


print('Best params:', model.best_params_)
print('CV score:',model.best_score_.round(3))
print('Test score:',model.score(dx_test,dy_test).round(3))