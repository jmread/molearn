#!/usr/bin/python

import sys

from numpy import *

set_printoptions(precision=3, suppress=True)

filename=sys.argv[1]
XY = genfromtxt(filename, skip_header=1, delimiter=",")
N,DL = XY.shape
L = int(sys.argv[2])
X = XY[:,L:DL]
Y = XY[:,0:L]

from molearn.core.tools import make_split
X_train,Y_train,X_test,Y_test = make_split(X,Y,split_percentage=0.50)

from sklearn import linear_model
h = linear_model.LogisticRegression()

from molearn.classifiers.BR import BR
br = BR(h)
from molearn.classifiers.CC import CC
cc = CC(h)

from time import clock
from molearn.core import metrics

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
