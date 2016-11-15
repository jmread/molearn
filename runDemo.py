#!/usr/bin/python

import sys
from time import *
from numpy import *
from molearn.core import metrics
from sklearn import linear_model
from molearn.classifiers.BR import BR
from molearn.classifiers.CC import CC

set_printoptions(precision=3, suppress=True)
basename = "molearn/data/"
dname="Music.csv"
L = 6
filename=basename+dname
XY = genfromtxt(filename, skip_header=1, delimiter=",")
N,DL = XY.shape
X = XY[:,L:DL]
Y = XY[:,0:L]
N_train = int(N/2)

X_train = X[0:N_train,:]
Y_train = Y[0:N_train,:]
X_test = X[N_train:N,:]
Y_test = Y[N_train:N,:]

h = linear_model.LogisticRegression()
br = BR(L,h)
cc = CC(L,h)

t0 = clock()
br.fit(X_train,Y_train)         
print("BR",)
print(metrics.Exact_match(Y_test,br.predict(X_test)),)
print(metrics.Hamming_loss(Y_test,br.predict(X_test)),)
print(clock() - t0)
t0 = clock()
cc.fit(X_train,Y_train)         
print("CC",)
print(metrics.Exact_match(Y_test,cc.predict(X_test)),)
print(metrics.Hamming_loss(Y_test,cc.predict(X_test)),)
print(clock() - t0)

from sklearn.tree import DecisionTreeClassifier
from molearn.classifiers.Ensemble import Ensemble
from molearn.classifiers.CC import RCC
ecc = Ensemble(base_estimator=RCC(DecisionTreeClassifier()),n_estimators=10)
t0 = clock()
ecc.fit(X_train,Y_train)         
print("ECC",)
print(metrics.Exact_match(Y_test,ecc.predict(X_test)),)
print(metrics.Hamming_loss(Y_test,ecc.predict(X_test)),)
print(clock() - t0)
