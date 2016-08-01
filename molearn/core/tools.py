from numpy import *

def make_split(X, Y, split_percentage=0.8):
    N,d = X.shape
    N,L = Y.shape
    N_train = int(N*split_percentage)
    N_test = N - N_train

    X_train = X[0:N_train,:]
    Y_train = Y[0:N_train,:]
    X_test = X[N_train:N,:]
    Y_test = Y[N_train:N,:]

    # make memory
    del X
    del Y

    return X_train,Y_train,X_test,Y_test 	

def make_XOR_dataset(n_tiles=1):
    # Teach network XOR function
    pat = array([
       # X X Y
        [0,0,0,0,0],
        [0,1,1,1,0],
        [1,0,1,1,0],
        [1,1,1,0,1]
    ],dtype=int)

    N,E = pat.shape
    D = 2
    L = E - D

    pat2 = zeros((N,E))
    pat2[:,0:L] = pat[:,D:E]
    pat2[:,L:E] = pat[:,0:D]
    pat2 = tile(pat2, (n_tiles, 1))
    random.shuffle(pat2)
    #print pat2

    Y = array(pat2[:,0:L],dtype=float)
    X = array(pat2[:,L:E],dtype=float)
#    savetxt("../data/XOR_.csv",pat2,fmt='%d',delimiter=",")
    return X, Y
