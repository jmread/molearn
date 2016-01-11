from numpy import *
from sklearn import linear_model

def phi(X, H, scale=0.2, drop=0.0):
    '''
    Generate new features ELM style 
    -------------------------------
    Justy apply a TLU (give between 0 and 1) on random weights.
    (The bias is added in case data is not normalised).

    X  : original label space
    H  : <number> of new features to generate
    drop : dropout rate
    scale : stochasticity around the threshold

    returns new matrix Z of new features
    '''
    N, D = X.shape
    B = (random.rand(H,D) > drop) * 1  # dropout
    W = random.randn(H,D) * B          # weights
    t = zeros(H)                       # biases (thresholds)
    
    # Trial (activation)
    A = dot(W,X.T).transpose()

    # Check biases & Var
    t = mean(A,axis=0)
    s = sqrt(var(A,axis=0))

    # New bias
    t = t  + random.randn(H)*scale*s

    # Apply TLU
    for i in range(N):
        A[i,:] = (A[i,:] > t) * A[i,:]

    return A,W,t


class ELM() :
    '''
        ELM = Extreme Learning Machine
        ---- 
        It just boosts the input with random feature functions, and then runs any MLC algorithm ontop of that.
    '''

    W = None    # Random weights
    b = None    # biases
    h = None    # base classifier

    L = -1      # Num labels
    H = 10      # Num hidden units



    def __init__(self, num_hidden=10, h=linear_model.LogisticRegression()):
        ''' hidden units h '''
        self.H = num_hidden
        self.h = h

    def train(self, X, Y):
        N,D = X.shape
        N,self.L = Y.shape

        # Boost into new space
        Z, self.W, self.b = phi(X, self.H, scale=0.2, drop=0.2)

        # Train
        self.h.train(Z,Y)

    def predict(self, X):
        N,D = X.shape

        # Boost into new space
        A = dot(self.W,X.T).transpose()
        for i in range(N):
            A[i,:] = (A[i,:] > self.b) * A[i,:]

        # Predict
        Y = self.h.predict(A)
        return Y

def demo():

    from tools import make_XOR_dataset
    from BR import BR

    X,Y = make_XOR_dataset()
    N,L = Y.shape

    h = linear_model.SGDClassifier(n_iter=100)
    nn = ELM(8,BR(L,h))
    nn.train(X, Y)
    # test it
    print nn.predict(X)
    print "vs"
    print Y

if __name__ == '__main__':
    demo()

