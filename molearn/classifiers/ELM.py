from numpy import *
from sklearn import linear_model

#####################################################################
# Note: This file contains both ELM (classifier) and ELR (regressor)
#####################################################################

def phi(X, H, f, scale=0.2, mask=0.0):
    '''
    Generate new features ELM style 
    -------------------------------
    1. Create random weights
    2. Apply activation function / non-linearity f
    3. Find the midpoint (mean), variance; make an appropriate offset/scale. 

    X  : original label space
    H  : num. of new features to generate
    mask : dropout rate
    scale : stochasticity around the threshold by this amount

    returns new matrix Z of new features
    '''
    N, D = X.shape
    B = (random.rand(H,D) > mask) * 1  # dropout
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
    Z = f(A)

    return Z,W,t


from molearn.core.functions import tlu, sigmoid, tanh

class ELM() :
    '''
        ELM = Extreme Learning Machine
        ---- 
        It just boosts the input with random feature functions, and then runs any MOP algorithm ontop of that.
        (Sholud work with either classification or regression)
    '''

    W = None    # Random weights
    b = None    # biases
    h = None    # base classifier

    L = -1      # Num labels
    H = 10      # Num hidden units

    density = 1.
    scale = 1.


    def __init__(self, num_hidden=10, h=linear_model.LogisticRegression(), f=tlu, density=1., scale=1.):
        ''' hidden units h '''
        self.H = num_hidden
        self.h = h
        self.f = f
        self.scale = scale
        self.density = density

    def fit(self, X, Y):
        N,D = X.shape
        N,self.L = Y.shape

        # Random Weights
        self.W = random.randn(self.H,D) * (random.rand(self.H,D) < self.density)
    
        # Activation
        A = dot(self.W,X.T).transpose()

        # Sensible Bias (mean plus some variation wrt standard deviation)
        self.b = mean(A,axis=0) + random.randn(self.H)*self.scale*sqrt(var(A,axis=0))

        # Apply TLU
        Z = self.f(A - self.b)
        print(Z)

        # Train
        self.h.fit(Z,Y)

        return self

    def predict(self, X):
        N,D = X.shape

        # Boost into new space
        A = dot(self.W,X.T).transpose() - self.b

        # Non-linearity
        Z = self.f(A)

        # Predict
        Y = self.h.predict(A)
        return Y

    def predict_proba(self, X):
        N,D = X.shape

        # Boost into new space
        A = dot(self.W,X.T).transpose() - self.b

        # Non-linearity
        Z = self.f(A)

        # Predict
        Y = self.h.predict_proba(A)
        return Y

class ELM_OI(ELM) :
    '''
        ELM_OI = Extreme Learning Machine, Output to input
        --------------------------------------------------
        It just boosts the OUTPUT (not the input as with ELMs) with random feature functions, 
        and then runs any MLR algorithm to these outputs ontop of that.

        see also iOLS.py
    '''

    def __init__(self, num_hidden=10, h=linear_model.LinearRegression()):
        ELM.__init__(self,num_hidden,h=h)

    def fit(self, X, Y):
        N,D = X.shape
        N,self.L = Y.shape

        self.W = random.randn(self.H,self.L)         # weights
        Wi = matrix(self.W).I                        # invert matrix
        Z = dot(Y,Wi)

        self.h.fit(X,Z)
        return self

    def predict(self, X):
        # Predict
        Z = self.h.predict(X)
        return dot(Z,self.W)

def demo():

    import sys
    sys.path.append( '../core' )
    from tools import make_XOR_dataset
    from BR import BR
    set_printoptions(precision=3, suppress=True)

    X,Y = make_XOR_dataset()
    N,L = Y.shape

    print("CLASSIFICATION")
    h = linear_model.SGDClassifier(n_iter=100)
    nn = ELM(8,f=tanh,h=BR(-1,h))
    nn.fit(X, Y)
    # test it
    print(nn.predict(X))
    print("vs")
    print(Y)

    print("REGRESSION")
    r = ELM(100,h=linear_model.LinearRegression())
    r.fit(X,Y)
    print(Y)
    print(r.predict(X))

    print("REGRESSION OI")
    r = ELM_OI(100,h=BR(-1,h=linear_model.SGDRegressor()))
    r.fit(X,Y)
    print(Y)
    print(r.predict(X))


if __name__ == '__main__':
    demo()

