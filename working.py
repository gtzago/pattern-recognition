# -*- coding: utf-8 -*-

from numpy.random import multivariate_normal
from os.path import os
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where

from pybrain.datasets import ClassificationDataSet
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain.structure.modules import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
from sklearn import preprocessing
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from cstr import ContinuouslyStirredTankReactor, pca, accuracy_fault_no_fault
import matplotlib.pyplot as plt
import numpy as np


########## Part 1 ##############
# path = os.path.join(os.path.dirname(__file__), 'results/')
# cstr = ContinuouslyStirredTankReactor(path)
# cstr.process_all(k=2)
# cstr.process_all(k=3)


####################### Part 2 #############################
# load the data.
path = os.path.join(os.path.dirname(__file__), 'results/selected_faults/')
cr = ContinuouslyStirredTankReactor(path)
#cr.load_data('4.csv')
cr.load_all()

x, y = cr.x, np.ravel(cr.y)
fault_types = np.union1d(y, y)
for i in range(0,fault_types.size):
    y[y==fault_types[i]]=i
    
n = y.size  # number of samples.
indexes = range(0, n)
np.random.shuffle(indexes)  # shuffles the samples.
ind_train = indexes[0:n / 2]  # half of samples for training.
ind_test = indexes[n / 2:]  # and other half for testing.

######################### PRE-PROCESSING #############################

# scales the data to have mean value equals 0 and standard deviation equals 1.
scaler = preprocessing.StandardScaler().fit(x[ind_train, :])

x_train = scaler.transform(x[ind_train, :])  # training features.
y_train = y[ind_train]  # training labels.
x_test = scaler.transform(x[ind_test, :])  # test features.
y_test = y[ind_test]  # test labels.

n_components = 2  # number of pca components to be used.
pca = PCA(n_components=n_components)
# generate the pca transformations based on the training features.
pca.fit(x_train)

x_train = pca.transform(x_train)  # applies the pca transformation.
x_test = pca.transform(x_test)

y_hat = {} # predicted outputs

####################################################

######### X used in decision region plotting ################
x_min, x_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
y_min, y_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

x_plt = np.c_[xx.ravel(), yy.ravel()]


############## SVM ################
print '\nTraining SVM'
lin_clf = svm.LinearSVC()  # creates a linear svm.
lin_clf.fit(x_train, y_train)  # trains the svm.
y_hat['svm'] = lin_clf.predict(x_test)


############## ANN ################
print '\nTraining Artificial Neural Network'
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
hiddenLayer = SigmoidLayer(15)
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

y_hat['ann'] = trainer.testOnClassData(dataset=tstdata, verbose=False,
                                return_targets=False)

################ Linear Machine ################
print '\nCalculating Linear Machine'
x_train_lm = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
x_test_lm = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

w = np.dot(np.linalg.pinv(x_train_lm), np.reshape(y_train, (-1, 1)))

y_hat['linear machine'] = np.dot(np.transpose(w), np.transpose(x_test_lm))
y_hat['linear machine'] = np.ravel(np.round(y_hat['linear machine']).astype(int))
aux = np.union1d(y_test,y_test)
y_hat['linear machine'][y_hat['linear machine']<aux[0]]=aux[0]
y_hat['linear machine'][y_hat['linear machine']>aux[-1]]=aux[-1]


for cls in y_hat.keys():
    c = confusion_matrix(y_test, y_hat[cls], labels=None)
    print '{:s}'.format(cls)
    print 'Confusion Matrix'
    print c
    print '\n'


######### Plotting decision regions ################

# SVM
plt.subplot(2,2,1)
z = lin_clf.predict(x_plt)
z = np.array(z)
z = z.reshape(xx.shape)
plt.contourf(xx, yy, z, alpha=0.4)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, alpha=0.8)
plt.title('SVM')

# ANN
pltdata = ClassificationDataSet(n_components)
for i in range(0, x_plt.shape[0]):
    pltdata.addSample(x_plt[i], [0])
z = trainer.testOnClassData(dataset=pltdata, verbose=False,
                                return_targets=False)
z = np.array(z)
z = z.reshape(xx.shape)

plt.subplot(2,2,2)
plt.contourf(xx, yy, z, alpha=0.4)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, alpha=0.8)
plt.title('Artificial Neural Network')


# linear machine
x_plt_lm = np.hstack((np.ones((x_plt.shape[0], 1)), x_plt))
z = np.dot(np.transpose(w), np.transpose(x_plt_lm))
z = np.ravel(np.round(z).astype(int))
z = np.array(z)
aux = np.union1d(y_test,y_test)
z[z<aux[0]]=aux[0]
z[z>aux[-1]]=aux[-1]
z = z.reshape(xx.shape)
plt.subplot(2,2,3)
plt.contourf(xx, yy, z, alpha=0.4)
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, alpha=0.8)
plt.title('Linear Machine')


plt.show()
