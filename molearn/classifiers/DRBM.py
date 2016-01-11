from numpy import *
from sklearn import linear_model
from BR import BR

class DRBM() :
    """
        DRBM
        -------------------
        Meta-RBM: Train an RBM, and then stick BR on top.
    """

    W = None

    H = 10
    L = -1

    h = None

    rbm = None

    def __init__(self, num_hidden=10, h=linear_model.LogisticRegression()):
        ''' for H hidden units, and base classifier 'h' '''
        self.H = num_hidden
        self.h = h

    def train(self, X, Y, W_e=None, E=500, lr=0.1):
        '''
            X: input
            Y: output
            h: base classifier
        '''
        # 0. Extract Dimensions
        N,D = X.shape
        self.L = Y.shape[1]

        # 1. RBM 

        from RBME import *
        self.rbm = RBM(num_visible = D, num_hidden = self.H, learning_rate = lr)
        self.rbm.train(X, max_epochs = E)
        Z = self.rbm.run_visible(X)

        #rng = random.RandomState(123)
        #from RBM import *
        #self.rbm = RBM(input=X,n_visible=D,n_hidden=self.H,numpy_rng=rng) 
        #for epoch in xrange(100):
        #    self.rbm.contrastive_divergence(lr=0.01, k=1)
        #A,Z = self.rbm.sample_h_given_v(X)

        #print Z 
        #from sklearn.neural_network import BernoulliRBM
        #self.rbm = BernoulliRBM(self.H, learning_rate = 0.01, n_iter = 10000)
        #self.rbm.fit(X)
        #Z = self.rbm.transform(X)
        #print Z 

        # 2. Train final layer, h : YFX -> Y
        self.h = BR(self.L,self.h)
        self.h.train(Z,Y)

    def predict(self, X):
        '''
            return predictions for X
        '''
        N,D = X.shape

        A = self.rbm.run_visible(X)
        #A,Z = self.rbm.sample_h_given_v(X)
        #print A,Z
        #Z = self.rbm.transform(X)

        # Propagate to final layer
        Y = self.h.predict(A > 0.5)
        return Y

def demo():
    from tools import make_XOR_dataset

    X,Y = make_XOR_dataset()
    N,L = Y.shape

    nn = DRBM(20)
    nn.train(X, Y)

    # test it
    print nn.predict(X)
    print "vs"
    print Y

if __name__ == '__main__':
    demo()

