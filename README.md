# sklearn
<br/>

## 1. 常用共通 import
| 目標 | 指令 | 說明 |
| --- | --- | --- |
|產生數據| from sklearn.datasets import make_blobs | 產生簇狀分布資料|
|| from sklearn.datasets import make_moons | 產生新月形分布資料|
|| from sklearn.preprocessing import StandardScaler| 執行標準化|
|| from sklearn.model_selection import train_test_split| 分割訓練資料|
|| import matplotlib.pyplot as plt ||
* 特徵資料的標準化：各特徵資料的範圍可能差異很大，使用標準化可以把所有特徵資料調整到固定範圍，加快機器學習模型的訓練速度、有機會提高預測準確率。
<br/>

## 2. 常用共通指令
| 目標 | 指令 | 說明 |
| --- | --- | --- |
|資料| `x`,`y`=make_moons(<br/>  n_samples=`資料比數`,<br/>n_features=`特徵數量`,<br/>centers=`標籤數量`,<br/>random_state=0) | 產生新月形分布資料<br/>x為特徵，y為標籤|
||`x`,`y`=make_blobs(<br/>  n_samples=`資料比數`,<br/>noise=`0.15`,<br/>random_state=0) | 產生簇狀分布資料<br/>x為特徵，y為標籤|
|| StandardScaler().fit_transform(`x_data`) | 資料標準化<br/>使特徵資料的平均數=0、變異數=1|
|| `x_train`,`x_test`,`y_train`,`y_test`=train_test_split(<br/>`data`,`label`,<br/>test_size=`0.2`,<br/>random_state=0) |分割訓練資料|
|模型|`模型名`.fit(`x_train`,`y_train`)|訓練模型|
||`predictions`= `模型名`.predict(`x_test`)|預測模型|
||`模型名`.score(`x_train`,`y_train`)<br/>`模型名`.score(`x_test`,`y_test`)|預測準確率|
<br/>

## 3. KNN (K-nearest neighbors)

- k個最近鄰居，選多數特徵決來分類
- KNN不需要訓練，稱為懶惰學習法
- K值選擇會影響預測結果與計算時間

| 目標 | 指令 | 說明 |
| --- | --- | --- |
|import|from sklearn.neighbors import KNeighborsClassifier||
|建立模型|`knn` = KNeighborsClassifier(n_neighbors=`5`)|尋找最近`5`筆鄰居資料，取多數特徵 |
<br/>

## 4. 邏輯斯回歸 (logistic regression)

- 將特徵資料轉換為0-1的值，來預測是否為某個標籤的機率
- 就分類結果而言，會在標籤之間畫出一條線性的決策邊界，是一種二元分類器
- 若資料本身很難完整一分為二，分類效果就會不佳

| 目標 | 指令 | 說明 |
| --- | --- | --- |  
|import|from sklearn.linear_model import LogisticRegression||
|建立模型|`log_reg` = LogisticRegression()||
<br/>

## 5. 線性支援向量機 (Linear SVM)

- SVM會將資料投射到更高維度、找出超平面，讓二維無法分類的資料找出分界線
- SVM分類效果比高邏輯斯回歸好，但也更費時；邏輯斯更適合處理大型資料集

| 目標 | 指令 | 說明 |
| --- | --- | --- | 
|import|from sklearn.svm import LinearSVC||
|建立模型| `linear_svm` = LinearSVC()||
<br/>

## 6. 非線性 SVM

- 當資料組成是線性不可分時使用
- 經過核函數進行維度轉換，來找到區分資料的超平面
- 從數學的角度來看，其實是線性的；只不過原始資料看不出來

| 目標 | 指令 | 說明 |
| --- | --- | --- | 
|import|from sklearn.svm import SVC||
|建立模型| `linear_svm` = SVC()||
<br/>
