from numpy import *
import copy
from sklearn.linear_model import LogisticRegression, SGDClassifier
from .CC import CC

#def score(self,P):
#    ''' score
#        -----------
#        We need to score predictions. This is exact match.
#    '''
#    return sum(log(P),axis=1)

#def likelihood(self,Y,P):
#    ''' 
#        A likelihood
#        ---------------
#        Y = the predictions   {0,1}
#        P = the probabilities [0,1]
#        so 1*1 is good.
#    '''
#    N,L = Y.shape
#    score = zeros((N,1))
#    for i in range(N):
#        for j in range(L):
#            score[i] = score[i] + (P[i,j]**Y[i,j]) * (1.-P[i,j])**(1.-Y[i,j])
#    return score

#def get_avg_of_samples(Y, P_Y):
#    f = zeros((N,self.L))
#    w_max = zeros(N)
#    for m in range(M):
#        w = self.likelihood(Y[:,m,:],P[:,m,:])
#        if w_m[n] < w_max[n]:
#            Y_max[n,:] = Y[n,:]

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
                y_ = array(map(int, binary_repr(b,width=self.L)))
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

    M = 100

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

    def predict(self, X, M = 10):
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
            # for M times
            for m in range(M):
                y_, w_ = self.sample_for_instance(X[n])
                # if it performs well, keep it, and record the max
                #print y_, w_
                if w_ > w_max:
                    Yp[n,:] = y_[:].copy()
                    w_max = w_

        return Yp


#
#    def sample(self,T):
#        '''
#            param T = 
#                    [ 0.3 0.7 ]      # example 1: [p(0|x), p(1|x)]
#                    [ 0.5 0.5 ]      # example 2: [p(0|x), p(1|x)]
#
#            returns 
#                * a column with indices weighted-sampled from T, and 
#                * a column of the respective probabilities of having sampled it
#
#                    y = [ 1
#                          0 ]
#                    p = [ 0.7
#                          0.5 ]
#                
#        '''
#        N,L = T.shape
#        y = zeros(N,dtype=int)
#        p_y = zeros(N)
#        for i in range(N):
#            y[i] = random.choice(L, 1, p=T[i,:]) 
#            p_y[i] = T[i,y[i]]
#        return y,p_y
#
#    def samples(self, X, M):
#        '''
#            Take M samples ~ p(y|X), e.g.,
#
#               y[1] y[2] y[3] p(y)
#               ----------------------
#                [1    0    1] 0.5
#                [1    0    0] 0.3
#               ----------------------
#        '''
#        N,D = X.shape
#
#        Y_max = zeros((N,self.L)) #CC.predict(self,X)
#        
#        X_ = zeros((N,D+self.L-1))
#        X_[:,0:D] = X
#
#        #samples = zeros((N,M,self.L))
#        Y = zeros((N,M,self.L))  # samples
#        P = zeros((N,M,self.L))
#        w = zeros((N,M))
#
#        for m in range(M):
#
#
#            # Get a sample for j = 1,2,3,...,L 
#            for j in range(self.L):
#
#                # SETUP XY
#                if j>0:
#                    X_[:,D+j-1] = Y[:,m,j-1]
#                    #X = column_stack([X, Y[:,j-1]])
#
#                # GET THE POSTERIOR PDF P(Y|XY) = [.,.,.,.] FOR ALL N EXAMPLES
#                P_Y = self.h[j].predict_proba(X_[:,0:D+j])
#                # SAMPLE FROM THE PROBABILITY
#                Y[:,m,j],P[:,m,j] = self.sample(P_Y)
#
#            w[:,m] = self.likelihood(Y[:,m,:],P[:,m,:])[:,0]
#
#        return Y,P,w
#
#
#
#
#    def paredict(self, X, M = -1):
#        '''
#            return predictions for X, taking M samples
#            if M == -1, use the default (self.M)
#        '''
#        if M < 0:
#            M = self.M
#
#        #return CC.predict(self,X)
#        Y,P,w = self.samples(X,M)
#
#        N,D = X.shape
#        Yreturn = zeros((N,self.L))
#        
#        for n in range(N):
#            Yreturn[n,:] = Y[n,argmax(w[n]),:]
#
#        return Yreturn

def demo():
    import sys
    sys.path.append("../")

    from core.tools import make_XOR_dataset

    X,Y = make_XOR_dataset(3)
    N,L = Y.shape

    print("PCC")
    pcc = PCC(L, SGDClassifier(n_iter=100,loss='log'))
    pcc.fit(X, Y)
    # test it
    Yp = pcc.predict(X[:,:])
    # evaluate it
    print(Yp[:,:])
    print("vs")
    print(Y[:,:])

    print("MCC")
    mcc = MCC(L, SGDClassifier(n_iter=100,loss='log'))
    mcc.fit(X, Y)
    Yp = mcc.predict(X[:,:], M=50)
    print(Yp[:,:])
    print("vs")
    print(Y[:,:])

    #from core.metrics import *
    #print metrics.Exact_match(Y,Yp), metrics.J_index(Y,Yp), metrics.Hamming_loss(Y,Yp)

if __name__ == '__main__':
    demo()

