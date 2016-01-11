import string
from time import *
from numpy import *
from functions import *

fudge_factor = 0.0000001 #for numerical stability

class BPNN:
    """
        BACK-PROPAGATION NEURAL NETWORK
        ------------------------------------------
    """

    def __init__(self, d, nh, no):

        self.H = nh   # number of hidden nodes
        self.L = no   # number of output nodes

        # create weights
        self.W_i = random.randn(d, self.H)
        self.b_i = random.randn(self.H) * 0.1
        self.W_o = random.randn(self.H, self.L)
        self.b_o = random.randn(self.L) * 0.1

    def forward_pass(self, X):
        '''
            Parameters
            ----------
            X: an n*d matrix of inputs 

            Method
            ---------
            1. make Z0: an n*(d+1) matrix (added bias)
            2. propagate (via W_i) to Z1
            3. propagate (via W_o) to Z2

        '''

        N,d = X.shape

        # Take input
        Z0 = X.copy()

        # Pass forward ... A = W'Z + b
        A = dot(Z0,self.W_i) + self.b_i
        Z1 = sigmoid(A)

        # Pass forward ... A = W'Z + b
        A = dot(self.W_o.transpose(),Z1.transpose()).transpose() + self.b_o
        Z2 = sigmoid(A)
       
        return Z0,Z1,Z2

    def back_propagate(self, X, Z, Y, T, N, M):
        """
        back_propagate
        --------------
        X: a n*d matrix
        Z: a n*h matrix
        Y: a n*L matrix
        T: a n*L matrix
        N: learning rate
        M: momentum
        """

        #============= GRADIENT ===========#

        # calculate error terms for output = E_o * sigmoid'(y)
        E_o = (T-Y)
        delta_o = E_o *  dsigmoid(Y)
        
        # calculate error terms for hidden
        E_i = dot(self.W_o,delta_o.transpose()).transpose()
        delta_i = E_i * dsigmoid(Z)

        #============= UPDATE =============#

        # update output weights
        grad = dot(delta_o.transpose(), Z).transpose()
        grad_ = E_o
        self.W_o = self.W_o  + N * grad
        self.b_o = self.b_o  + N * grad_

        # update input weights 
        grad = dot(delta_i.transpose(), X).transpose()
        grad_ = E_i
        self.W_i = self.W_i + N * grad
        self.b_i = self.b_i + N * grad_

    def predict(self, X):
        ''' 
        Forward Pass 
        ---------------
        '''
        X,Z,Y = self.forward_pass(X)
        return ((Y > 0.5) * 1.)

    def setWeights(self, W, b = None):
        ''' set weights W '''
        self.W_i = W.transpose()   # weights
        self.b_i = b.transpose()   # biases

    def print_weights(self):
        print('Input weights:')
        for i in range(self.D):
            print(self.W_i[i])
        print()
        print('Output weights:')
        for j in range(self.H):
            print(self.W_o[j])

    def train(self, X, Y, iterations=1000, N=0.5, M=0.1, batch_size=10, dropout = 0.):
        '''
            SGD
            --------------------------
            X: input
            Y: output
            iterations: of SGD
            N: learning rate
            M: momentum
            batch_size: batch size not yet implemented
            dropout: not yet implimented
        '''
        n,d = X.shape
        H,L = self.W_o.shape
        ''' select units to drop out randomly '''
        index = range(n)
        for t in range(iterations):

            random.shuffle(index)
            for i in index:
                Z0,Z1,Z2 = self.forward_pass(array([X[i,:]]))
                self.back_propagate(Z0,Z1,Z2,array([Y[i,:]]),N,M)

def demo():

    # Teach network XOR function
    pat = array([
       # X X Y
        [0,0,0,0,0],
        [0,1,1,1,0],
        [1,0,1,1,0],
        [1,1,1,0,1]
    ])

    D = 2
    E = 5
    L = E - D
    X = array(pat[:,0:D],dtype=float)
    Y = array(pat[:,D:E],dtype=float)
    print Y

    n = BPNN(D, 4, L)
    n.train(X, Y, iterations=1000, N=0.2, M=0.2)
    # test it
    print n.predict(X)


if __name__ == '__main__':
    demo()

