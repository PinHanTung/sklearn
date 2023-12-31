# sklearn 筆記

## 0. 使用流程
- 資料
  - 引入資料、確認資料：
    - 確認資料`head()`、行列數量`shape`、欄名與格式`info()`
    - 是否有NA或空值`isnull().sum()` `options.mode.use_inf_as_na = True`
  - 資料預處理：
    - 轉換資料格式(如要使用knn，需用LabelEncoder，將文字轉換成數字)
    - 資料裁切(如特徵與標籤為同個df，可使用iloc)
    - 分割資料集(train_test_split，分割後可以打印shape檢查)
- 模型
  - 建立模型(如有使用GridSearchCV尋找最佳參數，此時一同使用)
  - 訓練模型
  - 模型交叉驗證來修正參數(k-fold的cross_val_score) 此函數會修正嗎？
- 預測
  - 使用測試集預測
  - 計算訓練集準確率(score)
  - 計算交叉驗證準確率(val_score.mean()、或是有網格情況使用best_score_) 
  - 計算測試集準確率(score)
  - 此次最佳參數(mush.best_params_)
<br/>

## 1. 常用共通 import
| 目標 | 指令 | 說明 |
| --- | --- | --- |
|產生數據| from sklearn.datasets import make_blobs | 產生簇狀分布資料|
|| from sklearn.datasets import make_moons | 產生新月形分布資料|
|| from sklearn.datasets import load_iris| 讀取鳶尾花數據集|
|標準化| from sklearn.preprocessing import StandardScaler||
|model_select| from sklearn.model_selection import train_test_split| 分割訓練資料|
||from sklearn.model_selection import GridSearch| 網格搜尋，找到最佳參數並套入；最佳參數、測試集或交叉驗證準確率|
||from sklearn.model_selection import cross_val_score| k-fold交叉驗證，與其準確率|
|feature_select|from sklearn.feature_selection import SelectKBest, chi2| chi2為卡方統計，分類時尋找對預測有幫助的特徵；SelectKBest將結果最佳的欄挑選出來|
|預測結果報告| from sklearn.metrics import classification_report||
* 特徵資料的標準化：各特徵資料的範圍可能差異很大，使用標準化可以把所有特徵資料調整到固定範圍，加快機器學習模型的訓練速度、有機會提高預測準確率。
<br/>

## 2. 常用共通指令
| 目標 | 指令 | 說明 |
| --- | --- | --- |
|資料| `x`,`y`=make_moons(<br/>  n_samples=`資料比數`,<br/>n_features=`特徵數量`,<br/>centers=`標籤數量`,<br/>random_state=0) | 產生簇狀分布資料<br/>x為特徵，y為標籤|
||`x`,`y`=make_blobs(<br/>  n_samples=`資料比數`,<br/>noise=`0.15`,<br/>random_state=0) | 產生新月形分布資料<br/>x為特徵，y為標籤<br/>適合二元分類的非線性可分的情況|
|| StandardScaler().fit_transform(`x_data`) | 資料標準化<br/>使特徵資料的平均數=0、變異數=1|
|| `x`,`y`=load_iris(return_X_y=True)| 讀取鳶尾花數據集|
|| `x_train`,`x_test`,`y_train`,`y_test`=train_test_split(<br/>`data`,`label`,<br/>test_size=`0.2`,<br/>random_state=0) |分割訓練資料，前四個順序不能調換，須注意|
|模型|`模型名`.fit(`x_train`,`y_train`)|訓練模型|
||`predictions`= `模型名`.predict(`x_test`)|預測模型|
|報告|`模型名`.score(`x_train`,`y_train`)<br/>`模型名`.score(`x_test`,`y_test`)|預測準確率|
||classification_report(`y_test`, `predictions`)|通常看accuracy(整體精準率)、f1-score(precision,recall的調和平均數)|
<br/>

## 3-1. 模型1：KNN (K-nearest neighbors)

- k個最近鄰居，選多數特徵決來分類
- KNN不需要訓練，稱為懶惰學習法
- K值選擇會影響預測結果與計算時間
- 可使用for迴圈、或網格搜尋，尋找最佳k值
- 注意：knn資料須為數字，若為物件，需使用LabelEncoder()轉換

| KNN | 指令 | 說明 |
| --- | --- | --- |
|import|from sklearn.neighbors import KNeighborsClassifier||
|建立模型|`knn` = KNeighborsClassifier(n_neighbors=`5`)|尋找最近`5`筆鄰居資料，取多數特徵 |

| LabelEncoder | 指令 | 說明 |
| --- | --- | --- |
|import|from sklearn.preprocessing import LabelEncoder||
||`le`=LabelEncoder()|建立編碼器|
||`df`=`df`.apply(`le`.fit_transform)|fit_transform根據資料型態轉換並且回傳|

| GridSearch | 指令 | 說明 |
| --- | --- | --- |
|import|from sklearn.model_selection import GridSearchCV| 網格搜尋|
|建立模型|`model`=GridSearchCV(`KNeighborsClassifier()`,`套用參數的字典`)|找出最佳參數、套用到模型函式(然後再用fit訓練模型)|
|顯示結果|`model`.best_params|最佳參數|
||`model`.best_score_.round(3)|交叉驗證準確率(最佳參數)|
||`model`.score(`dx_test`,`dy_test`).round(3)|訓練集和測試集的預測準確率(最佳參數)|
<br/>

## 3-2. 模型2：邏輯斯回歸 (logistic regression)

- 將特徵資料轉換為0-1的值，來預測是否為某個標籤的機率
- 就分類結果而言，會在標籤之間畫出一條線性的決策邊界，是一種二元分類器
- 若資料本身很難完整一分為二，分類效果就會不佳

| 目標 | 指令 | 說明 |
| --- | --- | --- |  
|import|from sklearn.linear_model import LogisticRegression||
|建立模型|`log_reg` = LogisticRegression()||
<br/>

## 3-3. 模型3：線性支援向量機 (Linear SVM)

- SVM會將資料投射到更高維度、找出超平面，讓二維無法分類的資料找出分界線
- SVM分類效果比高邏輯斯回歸好，但也更費時；邏輯斯更適合處理大型資料集

| 目標 | 指令 | 說明 |
| --- | --- | --- | 
|import|from sklearn.svm import LinearSVC||
|建立模型| `linear_svm` = LinearSVC()||
<br/>

## 3-4. 模型4：非線性 SVM

- 當資料組成是線性不可分時使用
- 經過核函數進行維度轉換，來找到區分資料的超平面
- 從數學的角度來看，其實是線性的；只不過原始資料看不出來

| 目標 | 指令 | 說明 |
| --- | --- | --- | 
|import|from sklearn.svm import SVC||
|建立模型| `linear_svm` = SVC()||
<br/>


## 3-5. 模型5：決策樹 (decision tree)

- 樹狀的決策結構，根據特徵資料決定每個節點往哪走，最後得到預測標籤
- 因為對訓練資料十分敏感，可能會產生過度適配的問題，導致對新資料的預測準確率不佳

| 目標 | 指令 | 說明 |
| --- | --- | --- | 
|import|from sklearn.tree import DecisionTreeClassifier||
|建立模型| `tree` = DecisionTreeClassifier()||
<br/>


## 3-6. 模型6：隨機森林 (random forest)

- 由多個決策樹組成，會隨機將資料分給不同決策樹，藉此提高準確率
- 這種多個機器學習組合成的模型(可以是同一類或不同類)，稱為集成學習(ensemble learning)

| 目標 | 指令 | 說明 |
| --- | --- | --- | 
|import|from sklearn.ensemble import RandomForestClassifier||
|建立模型|`forest` = RandomForestClassifier()||
<br/>


## 4. 修正模型的超參數：k-fold交叉驗證法

- 除了訓練集和測試集，訓練過程中更嚴謹的做法還會分割出驗證資料集，用來修正模型的超參數
- 機器學習會需要幾回合，每回合結束會用驗證資料集查看準確度，來確認調整方向
- 若是從訓練集再切出驗證集，可以訓練的資料更少，所以使用k-fold交叉驗證法，把訓練集切成k等分，每個回合拉出一個部分當作驗證資料集，其他的為訓練集，這樣執行k回合

| 目標 | 指令 | 說明 |
| --- | --- | --- | 
|import|from sklearn.model_selection import cross_val_score||
|建立模型|`val_score` = cross_val_score(`model`, `x_train`, `y_train`, cv=`5`)|cv表示要切成幾等分，未指定則採預設值5|
|準確率|`val_score`.mean().round(3)|k-fold交叉驗證的平均準確率|
<br/>


## 5. 尋找最佳分類特徵：SelectKBest與chi2

| 目標 | 指令 | 說明 |
| --- | --- | --- | 
|import|from sklearn.feature_selection import SelectKBest, chi2| chi2為卡方統計，分類時尋找對預測有幫助的特徵；SelectKBest將結果最佳的欄挑選出來|
||`dx`=SelectKBest(chi2, k=`3`).fit_transform(`dx`, `dy`)|chi2為此次用來挑選的方法，k為挑選的特徵數量；得出的結果格式為ndarray，需要pd.DataFrame(`dx`)後再使用|
|chi2備註|`chi2_stat`, `p_values` = chi2(dx, dy)|第一個變數為卡方統計量，第二個變數為p值，p值越小、與標籤越相關|
