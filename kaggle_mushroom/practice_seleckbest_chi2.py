from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np

# 使用鳶尾花資料練習
X, y = load_iris(return_X_y=True)

# 使用selectkbest與chi卡方統計，找到對分類最有幫助的兩個特徵
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
# print('X.shape_before:',X.shape)
# print('X.shape_after:',X_new.shape)
# print(X[:5])
# print(X_new[:5])



# 計算卡方統計量和 p 值
chi2_stat, p_values = chi2(X, y)

# 顯示卡方統計量和 p 值
print("Chi2 Statistic:")
print(chi2_stat)

print("\nP-values:")
print(p_values)

# p值由小到大排序
print(np.sort(p_values))
