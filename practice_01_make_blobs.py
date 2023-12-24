from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dx, dy = make_blobs(n_samples=500, n_features=2, centers=2, random_state=0)
dx_std = StandardScaler().fit_transform(dx)

dx_train, dx_test, dy_train, dy_test = train_test_split(dx_std,dy,test_size=0.2,random_state=0)



#plt.scatter(dx_std.T[0],dx_std.T[1],c=dy,cmap='Dark2')

#plt.grid() 
#plt.show()

print(dx.shape)
print(dx_train.shape)
print(dx_test.shape)
print(dy.shape)
print(dy.train.shape)
print(dy_test.shape)