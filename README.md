# sklearn
<br/>

## 1. 參考指令
### 寫在最前面
| 目標 | 指令 | 說明 |
| --- | --- | --- |
|| from sklearn.datasets import make_blobs | make_blobs 用來產生數據集|
|| from sklearn.preprocessing import StandardScaler| StandardScaler 用來執行標準化|
|| import matplotlib.pyplot as plt ||
<br/>

## 2. 參考指令
| 目標 | 指令 | 說明 |
| --- | --- | --- |
|| `x`,`y`=make_blobs(n_samples=`資料比數`,n_features=`特徵數量`,centers=`標籤數量`,random_state=0) | 產生測試用資料，x為特徵，y為標籤|
|| StandardScaler().fit_transform(`x_data`) | 使資料標準化：使特徵資料的平均數=0、變異數=1|


特徵資料的標準化：
各特徵資料的範圍可能差異很大，使用標準化可以把所有特徵資料調整到固定範圍，加快機器學習模型的訓練速度、有機會提高預測準確率。
