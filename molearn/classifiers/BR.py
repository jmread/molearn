from numpy import *
import copy
from sklearn import linear_model

class BR() :

    h = None
    L = -1

    def __init__(self, L, h=linear_model.LogisticRegression()):
        self.L = L
        self.h = [ copy.deepcopy(h) for j in range(self.L)]

    def train(self, X, Y):
        #print "training ... [" ,
        for j in range(self.L):
            #print j , 
            self.h[j].fit(X, Y[:,j])
        #print "]"

    def predict(self, X):
        '''
            return predictions for X
        '''
        N,D = X.shape
        Y = zeros((N,self.L))
        for j in range(self.L):
            Y[:,j] = self.h[j].predict(X)
        return Y

def demo():
    from tools import make_XOR_dataset

    X,Y = make_XOR_dataset()
    N,L = Y.shape

    br = BR(L, linear_model.SGDClassifier(n_iter=100))
    br.train(X, Y)
    # test it
    print br.predict(X)
    print "vs"
    print Y

if __name__ == '__main__':
    demo()

