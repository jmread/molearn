from numpy import *
import copy
from sklearn import linear_model

class CC() :

    h = None
    L = -1

    def __init__(self, L, h=linear_model.LogisticRegression()):
        ''' todo: make copies of some h '''
        self.L = L
        self.h = [ copy.deepcopy(h) for j in range(self.L)]

    def train(self, X, Y):
        for j in range(self.L):
            if j>0:
                X = column_stack([X, Y[:,j-1]])
            self.h[j].fit(X, Y[:,j])

    def predict(self, X):
        '''
            return predictions for X
        '''
        N,D = X.shape
        Y = zeros((N,self.L))
        for j in range(self.L):
            if j>0:
                X = column_stack([X, Y[:,j-1]])
            Y[:,j] = self.h[j].predict(X)
        return Y

def demo():
    from tools import make_XOR_dataset

    X,Y = make_XOR_dataset()
    N,L = Y.shape

    cc = CC(L, linear_model.SGDClassifier(n_iter=100))
    cc.train(X, Y)
    # test it
    print cc.predict(X)
    print "vs"
    print Y

if __name__ == '__main__':
    demo()

