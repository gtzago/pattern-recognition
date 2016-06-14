# -*- coding: utf-8 -*-

from os.path import os

from sklearn import svm

from cstr import ContinuouslyStirredTankReactor, pca, accuracy_fault_no_fault
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

########## Part 1 ##############

# path = os.path.join(os.path.dirname(__file__), 'results/')
# cstr = ContinuouslyStirredTankReactor(path)
# cstr.process_all(k=2)
# cstr.process_all(k=3)

########## Part 2 ###############


# load the data.
path = os.path.join(os.path.dirname(__file__), 'results/')
cr = ContinuouslyStirredTankReactor(path)
cr.load_data('2.csv')
fault_type = 3
x, y = cr.x, np.ravel(cr.y)
y[y == fault_type] = 1

n = y.size  # number of samples.
indexes = range(0, n)
np.random.shuffle(indexes)  # shuffles the samples.
ind_train = indexes[0:n / 2]  # half of samples for training.
ind_test = indexes[n / 2:]  # and other half for testing.

# scales the data to have mean value equals 0 and standard deviation equals 1.
scaler = preprocessing.StandardScaler().fit(x[ind_test, :])

x_train = scaler.transform(x[ind_train, :])  # training features.
y_train = y[ind_train]  # training labels.
x_test = scaler.transform(x[ind_test, :])  # test features.
y_test = y[ind_test]  # test labels.

n_components = 3  # number of pca components to be used.
pca = PCA(n_components=n_components)
# generate the pca transformations based on the training features.
pca.fit(x_train)

x_train = pca.transform(x_train)  # applies the pca transformation.
x_test = pca.transform(x_test)

####################################################

############## SVM ################
lin_clf = svm.LinearSVC()  # creates a linear svm.
lin_clf.fit(x_train, y_train)  # trains the svm.

n_fault = y_test.size - np.count_nonzero(y_test)
prec, recall, fscore, support = precision_recall_fscore_support(
    lin_clf.predict(x_test), y_test)  # calculates the svm performance over the test data.

print 'SVM\n'
print 'Precision - {:3.2f}'.format(prec[np.argmax(support)])
print 'Recall - {:3.2f}'.format(recall[np.argmax(support)])
print 'F measure - {:3.2f}\n'.format(fscore[np.argmax(support)])

############## ANN ################
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal

# n = FeedForwardNetwork()
# inLayer = LinearLayer(2)
# hiddenLayer = SigmoidLayer(3)
# outLayer = LinearLayer(1)
#
# n.addInputModule(inLayer)
# n.addModule(hiddenLayer)
# n.addOutputModule(outLayer)
#
# in_to_hidden = FullConnection(inLayer, hiddenLayer)
# hidden_to_out = FullConnection(hiddenLayer, outLayer)
#
# n.addConnection(in_to_hidden)
# n.addConnection(hidden_to_out)
#
# n.sortModules()

# Pre processing

# x_aux = pca.transform(scaler.transform(x))
#
# alldata = ClassificationDataSet(n_components)
#
#
# for i in range(0, n):
#     alldata.addSample(x_aux[i], y[i])
# tstdata, trndata = alldata.splitWithProportion(0.25)

trndata = ClassificationDataSet(n_components)
for i in range(0, y_train.size):
    # add data to the pybrain structure.
    trndata.addSample(x_train[i], y_train[i])

tstdata = ClassificationDataSet(n_components)
for i in range(0, y_test.size):
    tstdata.addSample(x_test[i], y_test[i])

trndata._convertToOneOfMany()  # convert the label to multidimension label.
tstdata._convertToOneOfMany()

n = FeedForwardNetwork()
inLayer = LinearLayer(trndata.indim)
hiddenLayer = SigmoidLayer(10)
outLayer = LinearLayer(trndata.outdim)

n.addInputModule(inLayer)
n.addModule(hiddenLayer)
n.addOutputModule(outLayer)

in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

n.addConnection(in_to_hidden)
n.addConnection(hidden_to_out)

n.sortModules()

trainer = BackpropTrainer(
    n, dataset=trndata, momentum=0.1, verbose=False, weightdecay=0.01)
trainer.trainUntilConvergence(maxEpochs=100, verbose=False,
                              continueEpochs=6, validationProportion=0.25)


y_hat = trainer.testOnClassData(dataset=tstdata, verbose=False,
                                return_targets=False)

prec, recall, fscore, support = precision_recall_fscore_support(
    y_hat, y_test)  # calculates the ann performance over the test data.

print 'ANN\n'
print 'Precision - {:3.2f}'.format(prec[np.argmax(support)])
print 'Recall - {:3.2f}'.format(recall[np.argmax(support)])
print 'F measure - {:3.2f}\n'.format(fscore[np.argmax(support)])


################ Máquina Linear ##################

x_train_lm = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
x_test_lm = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

w = np.dot(np.linalg.pinv(x_train_lm), np.reshape(y_train, (-1, 1)))

y_hat = np.dot(np.transpose(w), np.transpose(x_test_lm))
y_hat = np.ravel(np.round(y_hat).astype(int))

prec, recall, fscore, support = precision_recall_fscore_support(
    y_hat, y_test)  # calculates the ann performance over the test data.

print 'Linear Machine\n'
print 'Precision - {:3.2f}'.format(prec[np.argmax(support)])
print 'Recall - {:3.2f}'.format(recall[np.argmax(support)])
print 'F measure - {:3.2f}\n'.format(fscore[np.argmax(support)])
# for i in range(20):
#     trainer.trainEpochs( 1 )
#     trnresult = percentError( trainer.testOnClassData(),
#                               trndata['class'] )
#     tstresult = percentError( trainer.testOnClassData(
#            dataset=tstdata ), tstdata['class'] )
#
#     print "epoch: %4d" % trainer.totalepochs, \
#           "  train error: %5.2f%%" % trnresult, \
#           "  test error: %5.2f%%" % tstresult


# fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )
# trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)
#
# for i in range(20):
#     trainer.trainEpochs( 1 )
#     trnresult = percentError( trainer.testOnClassData(),
#                               trndata['class'] )
#     tstresult = percentError( trainer.testOnClassData(
#            dataset=tstdata ), tstdata['class'] )
#
#     print "epoch: %4d" % trainer.totalepochs, \
#           "  train error: %5.2f%%" % trnresult, \
#           "  test error: %5.2f%%" % tstresult
#
#
# print prec

# w = lin_clf.coef_[0]
# a = -w[0] / w[1]
#
# xx = np.linspace(-5, 5)
# yy = a * xx - lin_clf.intercept_[0] / w[1]
#
# ww = lin_clf.coef_[0]
# wa = -ww[0] / ww[1]
# wyy = wa * xx - lin_clf.intercept_[0] / ww[1]
#
# # plot separating hyperplanes and samples
# h0 = plt.plot(xx, yy, 'k-', label='no weights')
# # h1 = plt.plot(xx, wyy, 'k--', label='with weights')
# plt.scatter(cr.x[:, 0], cr.x[:, 1], c=np.ravel(cr.y), cmap=plt.cm.Paired)
# plt.legend()
#
# plt.axis('tight')
# plt.show()


# LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
#      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#      multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
#      verbose=0)
# dec = lin_clf.decision_function([[1]])
# dec.shape[1]
