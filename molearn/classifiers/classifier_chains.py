from numpy import *
import copy
from sklearn.linear_model import LogisticRegression, SGDClassifier

class CC() :
    '''
        Classifier Chain

        N.B. The chain will be in 'default' order (in which labels appear in the dataset).
    '''

    h = None
    L = -1

    def __init__(self, h=LogisticRegression()):
        ''' init

            Parameters
            ----------
            h is the base classifier
        '''
        self.base_classifier = h

    def fit(self, X, Y):
        ''' fit
        '''
        N, self.L = Y.shape
        L = self.L
        N, D = X.shape

        self.h = [ copy.deepcopy(self.base_classifier) for j in range(L)]
        XY = zeros((N, D + L-1))
        XY[:,0:D] = X
        XY[:,D:] = Y[:,0:L-1]
        for j in range(self.L):
            self.h[j].fit(XY[:,0:D+j], Y[:,j])
        return self

    def partial_fit(self, X, Y):
        ''' partial_fit

            N.B. Assume that fit has already been called
            (i.e., this is more of an 'update')
        '''
        N, self.L = Y.shape
        L = self.L
        N, D = X.shape

        XY = zeros((N, D + L-1))
        XY[:,0:D] = X
        XY[:,D:] = Y[:,0:L-1]
        for j in range(L):
            self.h[j].partial_fit(XY[:,0:D+j], Y[:,j])

        return self

    def predict(self, X):
        ''' predict

            Returns predictions for X
        '''
        N,D = X.shape
        Y = zeros((N,self.L))
        for j in range(self.L):
            if j>0:
                X = column_stack([X, Y[:,j-1]])
            Y[:,j] = self.h[j].predict(X)
        return Y

    def predict_proba(self, X):
        ''' predict_proba

            Returns marginals [P(y_1=1|x),...,P(y_L=1|x,y_1,...,y_{L-1})] 
            i.e., confidence predictionss given inputs, for each instanec.

            N.B. This function suitable for multi-label (binary) data
                 only at the moment (may give index-out-of-bounds error if 
                 uni- or multi-target (of > 2 values) data is used in training).
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
        ''' fit 
            (shuffle the chain order first)
        '''
        N,L = Y.shape
        self.chain = arange(L)
        random.shuffle(self.chain)
        return CC.fit(self, X, Y[:,self.chain])

    def predict(self, X):
        ''' predict
            (and unshuffle the chain order afterwards)
        '''
        Y = CC.predict(self,X)
        return Y[:,argsort(self.chain)]


#def payoff(y,p_):
#    ''' payoff
#        
#        This is the standard payoff used in PCC. Others can be used.
#
#        Parameters
#        ----------
#        y = y[1],...,y[L]
#        p = p[1],...,p[L]
#
#        where P_x(Y_j=y[j])=p[j], according to an implied model. 
#
#        Returns
#        -------
#        The standard payoff (product).
#    '''
#    prd = 1. 
#    for j in range(len(p)):
#        prd = prd * (p[j] * y[j] + (1 - p[j]) * (1 - y[j]) )
#
#    return prd

def P(y, x, cc, payoff=prod):
    ''' Payoff function, P(Y=y|X=x)

        What payoff do we get for predicting y | x, under model cc.

        Parameters
        ----------
        x: input instance
        y: its true labels
        cc: a classifier chain
        payoff: payoff function

        Returns
        -------
        A single number; the payoff of predicting y | x.
    '''
    D = len(x)
    L = len(y)

    p = zeros(L)
    xy = zeros(D + L)
    xy[0:D] = x.copy()
    for j in range(L):
        P_j = cc.h[j].predict_proba(xy[0:D+j].reshape(1,-1))[0] # e.g., [0.9, 0.1] wrt 0, 1
        xy[D+j] = y[j]                                            # e.g., 1
        p[j] = P_j[y[j]]                                          # e.g., 0.1
                                                                  #   or, y[j] = 0 is predicted with probability p[j] = 0.9
    return payoff(p)

class PCC(CC):
    '''
        Probabilistic Classifier Chains (PCC)
    '''

    def predict(self, X):
        ''' Predict

            Explores all possible branches of the probability tree.
            (i.e., all possible 2^L label combinations).

            Returns
            -------
            Predictions Y.
        '''
        N,D = X.shape
        Yp = zeros((N,self.L))

        # for each instance
        for n in range(N):
            w_max = 0.
            # for each and every possible label combination
            for b in range(2**self.L):
                # put together a label vector
                y_ = array(list(map(int, binary_repr(b,width=self.L))))
                # ... and gauge a probability for it (given x)
                w_ = P(y_,X[n],self)
                # if it performs well, keep it, and record the max
                if w_ > w_max:
                    Yp[n,:] = y_[:].copy()
                    w_max = w_

        return Yp


class MCC(CC):
    ''' Monte Carlo Sampling Classifier Chains
        
        PCC, using Monte Carlo sampling, published as 'MCC'.
        M samples are taken from the posterior distribution.

        N.B. Multi-label (binary) only at this moment.
    '''

    M = 10

    def __init__(self, h=LogisticRegression(), M=10):
        ''' Do M iterations, unless overridded by M at predict()tion time '''
        CC.__init__(self,h)
        self.M = M

    def sample(self, x):
        '''
            Sample y ~ P(y|x)

            Returns
            -------
            y: a sampled label vector
            p: the associated probabilities, i.e., p(y_j=1)=p_j
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

        return y, p

    def predict(self, X, M = 'default'):
        ''' Predict

            Parameters
            ----------
            X: Input matrix, (an Numpy.ndarray of shape (n_samples, n_features)
            M: Number of sampling iterations

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
            w_max = P(Yp[n,:].astype(int),X[n],self)
            # for M times
            for m in range(M):
                y_, p_ = self.sample(X[n]) # N.B. in fact, the calcualtion p_ is done again in P.
                w_ = P(y_.astype(int),X[n],self)
                # if it performs well, keep it, and record the max
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
    print("with 50 iterations ...")
    print(Yp)
    Yp = mcc.predict(X, 'default')
    print("with default (%d) iterations ..." % 1000)
    print(Yp)

    print("PCC")
    pcc = PCC(SGDClassifier(n_iter=100,loss='log'))
    pcc.fit(X, Y)
    print(pcc.predict(X))



if __name__ == '__main__':
    demo()

