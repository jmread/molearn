from numpy import *
import copy
from sklearn.linear_model import LogisticRegression, SGDClassifier

class CC() :
    '''
        Classifier Chain
        ----------------
        The chain will be in 'default' order (in which labels appear in the dataset)
    '''

    h = None
    L = -1

    def __init__(self, h=LogisticRegression()):
        '''
            h is the base classifier
        '''
        self.hop = h

    def fit(self, X, Y):
        N, self.L = Y.shape
        L = self.L
        N, D = X.shape

        self.h = [ copy.deepcopy(self.hop) for j in range(L)]
        XY = zeros((N, D + L-1))
        XY[:,0:D] = X
        XY[:,D:] = Y[:,0:L-1]
        for j in range(self.L):
            self.h[j].fit(XY[:,0:D+j], Y[:,j])
        return self

    def partial_fit(self, X, Y):
        '''
            assume that fit has already been called
            (i.e., this is more of an 'update')
        '''
        N, self.L = Y.shape
        L = self.L
        N, D = X.shape

        XY = zeros((N, D + L-1))
        XY[:,0:D] = X
        XY[:,D:] = Y[:,0:L-1]
        for j in range(L):
            self.h[j].fit(XY[:,0:D+j], Y[:,j])

        return self

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

    def predict_proba(self, X):
        '''
            return confidence predictions for X
            WARNING: for multi-label (binary) distribution only at the moment.
                ( may give index-out-of-bounds error if uni- or multi-target (of > 2 values) data is used in training )
        '''
        N,D = X.shape
        Y = zeros((N,self.L))
        for j in range(self.L):
            if j>0:
                X = column_stack([X, Y[:,j-1]])
            Y[:,j] = self.h[j].predict_proba(X)[:,1]
        return Y


class RCC(CC):
    '''
        Random Classifier Chain
        -----------------------
        Chain will be in a random order.
    '''

    chain = None

    def fit(self, X, Y):
        N,L = Y.shape
        self.chain = arange(L)
        random.shuffle(self.chain)
        return CC.fit(self, X, Y[:,self.chain])

    def predict(self, X):
        '''
            return predictions for X
        '''
        Y = CC.predict(self,X)
        return Y[:,argsort(self.chain)]


class PCC(CC):
    '''
        Probabilistic Classifier Chains (PCC)
        -------------------------------------
        
        Bayes-Optimal Inference
    '''

    def probability_for_instance(self, x, y):
        '''
            Return P(Y=y|X=x)
        '''
        D = len(x)
        L = len(y)

        p = zeros(L)
        xy = zeros(D + L - 1)
        xy[0:D] = x.copy()
        xy[D:] = y[0:L-1].copy()

        for j in range(L):
            p[j] = self.h[j].predict_proba(xy[0:D+j].reshape(1,-1))[0][y[j]]

        return prod(p)

    def predict(self, X):
        '''
            Predict
            -------
            return predictions for X
        '''
        N,D = X.shape
        Yp = zeros((N,self.L))

        # for each instance
        for n in range(N):
            w_max = 0.
            #print "--", X[n], "--"
            # for each and every possible label combination
            for b in range(2**self.L):
                # put together a label vector
                y_ = array(list(map(int, binary_repr(b,width=self.L))))
                # ... and gauge a probability for it (given x)
                w_ = self.probability_for_instance(X[n],y_)
                # if it performs well, keep it, and record the max
                #print y_, w_
                if w_ > w_max:
                    Yp[n,:] = y_[:].copy()
                    w_max = w_

        return Yp


class MCC(CC):
    '''
        Probabilistic Classifier Chains (PCC)
        -------------------------------------
        
        PCC, using Monte Carlo sampling, published as 'MCC'.
        M samples are taken from the posterior distribution.

        N.B. Multi-label (binary) only at this moment.
    '''

    M = 10

    def __init__(self, h=LogisticRegression(), M=10):
        ''' Do M iterations, unless overridded by M at predict()tion time '''
        CC.__init__(self,h)
        self.M = M

    def prob_for_instance(self, x, y):
        '''
            As for sample_for_instance but not sampling :-) 
            (using y instead)
        '''
        D = len(x)

        p = zeros(self.L)
        xy = zeros(D + self.L)
        xy[0:D] = x.copy()
        for j in range(self.L):
            P_j = self.h[j].predict_proba(xy[0:D+j].reshape(1,-1))[0]
            y_j = y[j]
            xy[D+j] = y_j
            p[j] = P_j[y_j]
        return y, prod(p)

    def sample_for_instance(self, x):
        '''
            Return P(Y=y|X=x)
        '''
        D = len(x)

        p = zeros(self.L)
        y = zeros(self.L)
        xy = zeros(D + self.L)
        xy[0:D] = x.copy()

        for j in range(self.L):
            P_j = self.h[j].predict_proba(xy[0:D+j].reshape(1,-1))[0]
            y_j = random.choice(2,1,p=P_j)
            xy[D+j] = y_j
            y[j] = y_j
            p[j] = P_j[y_j]

        return y, prod(p)

    def predict(self, X, M = 'default'):
        '''
            Predict
            -------
            NB: quite similar to PCC's predict function.
            Depending on the implementation, y_max, w_max may be initially set to 0, 
            if we wish to rely solely on the sampling. Setting the w_max based on a naive CC prediction
            gives a good baseline to work from.

            return predictions for X
        '''
        N,D = X.shape
        Yp = zeros((N,self.L))

        if M == 'default':
            M = self.M

        # for each instance
        for n in range(N):
            Yp[n,:] = CC.predict(self, X[n].reshape(1,-1))
            y_max, w_max = self.prob_for_instance(X[n],Yp[n,:].astype(int))
            # for M times
            for m in range(M):
                y_, w_ = self.sample_for_instance(X[n])
                # if it performs well, keep it, and record the max
                #print y_, w_
                if w_ > w_max:
                    Yp[n,:] = y_[:].copy()
                    w_max = w_

        return Yp


#class GCC(CC):
#
#    ''' Generalized CC '''
#
#    def __init__(self, h=LogisticRegression(), inference="greedy"):
#        '''
#            h is the base classifier
#        '''
#        self.hop = h



def demo():
    import sys
    from molearn.core.tools import make_XOR_dataset

    X,Y = make_XOR_dataset()
    N,L = Y.shape

    print(Y)
    print("vs")

    print("RCC")
    cc = RCC(SGDClassifier(n_iter=100,loss='log'))
    cc.fit(X, Y)
    print(cc.predict(X))

    print("MCC")
    mcc = MCC(SGDClassifier(n_iter=100,loss='log'),M=1000)
    mcc.fit(X, Y)
    Yp = mcc.predict(X, M=50)
    print(Yp)
    Yp = mcc.predict(X, 'default')
    print(Yp)

    print("PCC")
    pcc = PCC(SGDClassifier(n_iter=100,loss='log'))
    pcc.fit(X, Y)
    print(pcc.predict(X))



if __name__ == '__main__':
    demo()

