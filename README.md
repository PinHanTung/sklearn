# sklearn
<br/>

## 1. 參考指令
### 寫在最前面
| 目標 | 指令 | 說明 |
| --- | --- | --- |
|| from sklearn.datasets import make_blobs | 產生數據集|
|| from sklearn.preprocessing import StandardScaler| 執行標準化|
|| from sklearn.model_selection import train_test_split| 分割訓練資料|
|| import matplotlib.pyplot as plt ||
<br/>

## 2. 參考指令
| 目標 | 指令 | 說明 |
| --- | --- | --- |
| 產生測試用資料 | `x`,`y`=make_blobs(<br/>  n_samples=`資料比數`,<br/>n_features=`特徵數量`,<br/>centers=`標籤數量`,<br/>random_state=0) | x為特徵，y為標籤|
| 資料標準化 | StandardScaler().fit_transform(`x_data`) | 使特徵資料的平均數=0、變異數=1|
| 分割訓練資料 | `x_train`,`x_test`,`y_train`,`y_test`=train_test_split(<br/>`data`,`label`,<br/>test_size=`0.2`,<br/>random_state=0) ||


* 特徵資料的標準化：各特徵資料的範圍可能差異很大，使用標準化可以把所有特徵資料調整到固定範圍，加快機器學習模型的訓練速度、有機會提高預測準確率。
