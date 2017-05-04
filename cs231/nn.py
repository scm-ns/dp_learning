import numpy as np
import matplotlib.pyplot as plt


N = 100
D = 2
K = 3
X = np.zeros((N*K, D))
y = np.zeros(N*K , dtype='uint8')

for cls in range(K):
    ix = range(N*cls , N*(cls + 1))
    radius = np.linspace(0.0 , 1 , N)
    theta = np.linspace(cls * 4 , (cls + 1) * 4 , N) + np.random.randn(N) * 0.2
    X[ix] = np.c_[radius * np.sin(theta) , radius * np.cos(theta)]
    y[ix] = cls

plt.scatter(X[: ,0] , X[: , 1], c = y , s = 40 , cmap = plt.cm.Spectral)
plt.show()

W = 0.01 * np.random.randn(D , K)
b = np.zeros((1, K))

scores = np.dot(X , W) + b 
print(scores)



