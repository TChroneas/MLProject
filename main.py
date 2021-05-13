import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def gradcheck_softmax(W1init,W2init, X, t, lamda):
    W1 = np.random.rand(*W1init.shape)
    W2 = np.random.rand(*W2init.shape)
    epsilon = 1e-6

    _list = np.random.randint(X.shape[0], size=5)
    x_sample = np.array(X[_list, :])
    t_sample = np.array(t[_list, :])

    Ew, gradEw2 = cost_grad_softmax(W1,W2, x_sample, t_sample, lamda)

    print("gradEw shape: ", gradEw2.shape)

    numericalGrad = np.zeros(gradEw2.shape)
    # Compute all numerical gradient estimates and store them in
    # the matrix numericalGrad
    for k in range(numericalGrad.shape[0]):
        for d in range(numericalGrad.shape[1]):
            # add epsilon to the w[k,d]
            w_tmp = np.copy(W2)
            w_tmp[k, d] += epsilon
            e_plus, _ = cost_grad_softmax(W1,w_tmp, x_sample, t_sample, lamda)

            # subtract epsilon to the w[k,d]
            w_tmp = np.copy(W2)
            w_tmp[k, d] -= epsilon
            e_minus, _ = cost_grad_softmax(W1,w_tmp, x_sample, t_sample, lamda)

            # approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            numericalGrad[k, d] = (e_plus - e_minus) / (2 * epsilon)

    return (gradEw2, numericalGrad)
def load_data():


    # load the train files
    df = None

    y_train = []

    for i in range(10):
        tmp = pd.read_csv('data/mnist/train%d.txt' % i, header=None, sep=" ")
        # build labels - one hot vector
        hot_vector = [1 if j == i else 0 for j in range(0, 10)]

        for j in range(tmp.shape[0]):
            y_train.append(hot_vector)
        # concatenate dataframes by rows
        if i == 0:
            df = tmp
        else:
            df = pd.concat([df, tmp])

    train_data = df.to_numpy()
    y_train = np.array(y_train)

    # load test files
    df = None

    y_test = []



    for i in range(10):
        tmp = pd.read_csv('data/mnist/test%d.txt' % i, header=None, sep=" ")
        # build labels - one hot vector

        hot_vector = [1 if j == i else 0 for j in range(0, 10)]

        for j in range(tmp.shape[0]):
            y_test.append(hot_vector)
        # concatenate dataframes by rows
        if i == 0:
            df = tmp
        else:
            df = pd.concat([df, tmp])

    test_data = df.to_numpy()
    y_test = np.array(y_test)

    return train_data, test_data, y_train, y_test
def calcz(x):
    z= h3(x)
    z = np.hstack((np.ones((z.shape[0], 1)), z))
    return z
def calcdz(x):
    z=h3d(x)
    z = np.hstack((np.ones((z.shape[0], 1)), z))
    return z
def h1(a):
    return np.log(1 + np.exp(a))
def h2(a):
    return (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))
def h3(a):
    return np.cos(a)
def h1d(a):
    return (np.exp(a))/(np.exp(a)+1)
def h2d(a):
    return 1-np.square(np.tanh(a))
def h3d(a):
    return -np.sin(a)
def softmax(x, ax=1):
    m = np.max(x, axis=ax, keepdims=True)  # max per row
    p = np.exp(x - m)
    return p / np.sum(p, axis=ax, keepdims=True)
def cost_grad_softmax(W1,W2, X, t, lamda):
    Z=(calcz(X.dot(W1.T)))
    Zd=calcdz(X.dot(W1.T))
    Y = softmax(Z.dot(W2.T))
    max_error = np.max(Y, axis=1)
    # Compute the cost function to check convergence
    # Using the logsumexp trick for numerical stability - lec8.pdf slide 43
    Ew = np.sum(t * Y) - np.sum(max_error) - \
         np.sum(np.log(np.sum(np.exp(Y - np.array([max_error, ] * Y.shape[1]).T), 1))) - \
         (0.5 * lamda) * (np.sum(np.square(W1))+np.sum(np.square(W2)))

    # calculate gradient
    gradEw2 = (t - Y).T.dot(Z) - lamda * W2
  ##  gradEw1=(t -Y)*W2
    gradEw1=((t-Y).dot(W2)*Zd).T[:][1:].dot(X)-lamda*W1
   ## print(W1.shape)


    return Ew, gradEw2,gradEw1
def ml_softmax_train(t, X, lamda, W1init,W2init, options):
    W1 = W1init
    W2 = W2init
    _iter = options[0]
    tol = options[1]
    eta = options[2]
    batch_size=options[3]
    Ewold = -np.inf
    costs = []
    for i in range(1, _iter + 1):
        X,t=shuffle(X,t)
        for j in range(0, X.shape[0], batch_size):

             X_mini = X[j:j + batch_size]
             t_mini = t[j:j + batch_size]


        Ew, gradEw2,gradEw1 = cost_grad_softmax(W1,W2, X_mini, t_mini, lamda)
        costs.append(Ew)

        print('Iteration : %d, Cost function :%f' % (i, Ew))
        if np.abs(Ew - Ewold) < tol:
            break
        W2 = W2 + eta * gradEw2
        W1= W1+eta*gradEw1


        Ewold = Ew

    return W2,W1, costs




def ml_softmax_test(W1,W2, X_test):
    z=calcz(X_test.dot(W1.T))
    y=softmax(z.dot(W2.T))
    ttest=np.argmax(y,1)
    return ttest
def main():
    M = 200
    K=10
    X_train, X_test, y_train, y_test = load_data()
    X_train = X_train.astype(float) / 255
    X_test = X_test.astype(float) / 255
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    lamda = 0.01
    N, D = X_train.shape
    print(D)
    print(X_test.shape,y_test.shape)
    W1 = np.random.rand(M, D) * np.sqrt(2. / (D + M))
    W2 = np.random.rand(K, M + 1) * np.sqrt(2. / (M + 1 + K))
    options = [500, 1e-6, 0.5/200,200]
    W2,W1,costs =ml_softmax_train(y_train,X_train,lamda,W1,W2,options)
    pred=ml_softmax_test(W1,W2,X_test)
    print(np.mean(pred==np.argmax(y_test,1)))


if __name__ == "__main__":
    main()