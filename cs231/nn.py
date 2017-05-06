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

reg = 1e-3
step_size = 1e-0

for i in xrange(200):
    scores = np.dot(X , W) + b 
    print(scores)

    ## SOFTMAX applied to each prediction
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum( exp_scores , axis = 1 , keepdims = True)

    ## SOFTMAX CLASSIFIER LOSS Function
    correct_logprob = -np.log(probs[range(N * K) , y])
    data_loss= np.sum(correct_logprob) / (N*K)
    reg_loss = 0.5 * reg * np.sum(W * W);
    loss = data_loss = reg_loss

    if i % 10 == 0:
        print "iteration %d : loss %f " , (i , loss)

    grad_scores = probs 
    grad_scores[range(N*K) , y] -= 1
    grad_scores /= (N*K)

    dW = np.dot(X.T , grad_scores)
    db = np.sum(grad_scores , axis = 1 , keepdims = True)
    dW += reg * W

    W += - step_size * dW
    b += - step_size * db


scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print 'training accuracy: %.2f' % (np.mean(predicted_class == y))


