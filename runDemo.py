#!/usr/bin/python

import sys
from molearn.core import metrics
from time import *
from numpy import *
from sklearn import linear_model
from molearn.classifiers.BR import BR
from molearn.classifiers.CC import CC

set_printoptions(precision=3, suppress=True)
filename=sys.argv[1]
XY = genfromtxt(filename, skip_header=1, delimiter=",")
N,DL = XY.shape
L = int(sys.argv[2])
X = XY[:,L:DL]
Y = XY[:,0:L]
N_train = N/2

X_train = X[0:N_train,:]
Y_train = Y[0:N_train,:]
X_test = X[N_train:N,:]
Y_test = Y[N_train:N,:]

h = linear_model.LogisticRegression()
br = BR(L,h)
cc = CC(L,h)

t0 = clock()
br.fit(X_train,Y_train)         
print "BR",
print metrics.Exact_match(Y_test,br.predict(X_test)),
print metrics.Hamming_loss(Y_test,br.predict(X_test)),
print clock() - t0
t0 = clock()
cc.fit(X_train,Y_train)         
print "CC",
print metrics.Exact_match(Y_test,cc.predict(X_test)),
print metrics.Hamming_loss(Y_test,cc.predict(X_test)),
print clock() - t0
