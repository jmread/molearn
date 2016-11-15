from numpy import *
from scipy.stats import mode
import copy

from sklearn.ensemble import BaggingClassifier

class Ensemble(BaggingClassifier):
    '''
        A Simple Ensemble 
        ------------------
        This class should eventually not be needed, as we could use sklearn.ensemble.BaggingClassifier,
        but at the time of coding, this option doesn't work for molearn multi-label classifiers.

        In particular, note that this method is designed for the multi-dimensional output case:
            each 'label' may take more than one value (not necessarily just binary).
    '''

    h = None
    M = 10
    L = -1

    def __init__(self, L=-1, base_estimator=None, n_estimators=10):
        ''' note: L option to be deprecated here ! '''
        self.M = n_estimators
        self.h = [ copy.deepcopy(base_estimator) for m in range(self.M) ]

    def fit(self, X, Y):
        '''
            Simply fit each model individually.
        '''
        N,self.L = Y.shape
        for m in range(self.M):
            self.h[m].fit(X,Y)
        return self

    def predict(self, X):
        '''
            return predictions for X 
            (multi-dimensionally speaking, i.e., we return the mode)
        '''
        N,D = X.shape
        Y = zeros((N,self.L))
        for i in range(N):
            V = zeros((self.M,self.L))
            for m in range(self.M):
                V[m,:] = self.h[m].predict(array([X[i,:]]))
            Y[i,:] = mode(V)[0]
        return Y

    def predict_proba(self,X):
        '''
            return confidences (i.e., p(y_j|x)) 
            (in the multi-dimensional output case, this should be an N x L x K array
            but @NOTE/@TODO: this is not the case at the moment! At the moment it is N x L x 2;
            For example, in 
                [[ 0.   1. ]
                 [ 0.   0.9]
                 [ 0.   1. ]
                 [ 0.   1. ]
                 [ 0.   1. ]
                 [ 1.   0.9]]
                y_j=6 with probability 0.9.
            )
        '''
        N,D = X.shape
        Y = zeros((N,self.L,2))
        for i in range(N):
            V = zeros((self.M,self.L))
            for m in range(self.M):
                V[m,:] = self.h[m].predict(array([X[i,:]]))
            k = mode(V)[0]
            Y[i,:,0] = k
            Y[i,:,1] = sum(V==k,axis=0)/self.M
        return Y

def demo():
    import sys
    sys.path.append( '../core' )
    from tools import make_XOR_dataset

    X,Y = make_XOR_dataset()
    N,L = Y.shape

    from sklearn import linear_model
    h_ = linear_model.SGDClassifier(n_iter=100)
    from CC import RCC
    cc = RCC(h=h_)
    e = Ensemble(n_estimators=10,base_estimator=cc)
    e.fit(X, Y)
    # test it
    print(e.predict(X))
    print("vs")
    print(Y)

if __name__ == '__main__':
    demo()


