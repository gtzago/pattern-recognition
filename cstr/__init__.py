# -*- coding: utf-8 -*-

import os

from matplotlib import lines

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class ContinuouslyStirredTankReactor(object):

    def __init__(self, path):
        self.path = path
        self.x = None
        self.y = None

    def load_data(self, csv_file):
        '''
        Acessa a base de dados Jain localizada em CSV_FILE.
        '''
        import csv

        csv_file = os.path.join(self.path, csv_file)

        x = np.array([])
        y = np.array([])
        with open(csv_file, 'rb') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=';', quotechar='|')
            for row in csvreader:
                if len(row) > 1:
                    aux = [float(i) for i in row[0:-1]]
                    x = np.append(x, aux)
                    if row[-1].strip() == 'normal':
                        y = np.append(y, 0)
                    else:
                        y = np.append(y, int(row[-1]))

        (self.x, self.y) = np.reshape(x, (-1, 18)), np.reshape(y, (-1, 1))

        
    def load_all(self):
        '''
            Load all csv data in the directory.
        '''
        x = None
        y = None
        
        for fn in os.listdir(self.path):
            if os.path.isfile(os.path.join(self.path, fn)) and fn[-3:]=='csv':
                self.load_data(fn)
                if x is None:
                    x = np.copy(self.x)
                    y = np.copy(self.y)
                else:
                    x = np.vstack((x,self.x))
                    y = np.vstack((y,self.y))
        self.x = np.copy(x)
        self.y = np.copy(y)

    def process_all(self, k=3):
        for fn in os.listdir(self.path):
            if os.path.isfile(os.path.join(self.path, fn)):
                try:
                    self.load_data(fn)
                    if k > 3:
                        self.plot_pca(k=3, name=fn[:-4])
                    else:
                        self.plot_pca(k, name=fn[:-4])
                    print '\nPrinting ' + fn + ' data.'

                    yhat, t2 = self.calculate_t2_statistics(k, name=fn[:-4])

                    acc = accuracy_fault_no_fault(self.y, yhat)

                    print 'Acurácia utilizando apenas a estatística t^2 em {:d} principais componentes: {:f} %'.format(k, 100 * acc)

                except:
                    pass

    def plot_pca(self, k=2, name='scatter'):
        '''
            Plot the PCA with k components.
        '''
        mu = np.mean(self.x)
        var = np.var(self.x)

        index = np.nonzero(self.y == 0)
        x_normal = self.x[index[0], :]

        mu_normal = np.mean(x_normal, 0)
        sd = np.std(x_normal, 0)

        self.x = (self.x - mu_normal) / sd  # normaliza os dados

        xhat, wl, vl = pca(self.x, k)  # dados na nova base

        fault_classes = np.union1d(self.y, self.y)

        x_dict = {}

        for cls in fault_classes:
            index = np.nonzero(self.y == cls)
            x_cls = self.x[index[0], :]
            # dicionario com as amostras separadas por classe.
            x_dict[cls] = np.copy(x_cls)

        xhat_dict = {}  # dicionario que ira manter os novos dados por classe
        # cores para a plotagem
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', ]
        i = 0
        if k == 2:
            plt.Figure
            plt.hold(True)
            h = []
            legenda = []
            for cls in x_dict.keys():
                x_cls = x_dict[cls]
                xhat_dict[cls] = np.real(
                    np.dot((x_cls - mu) / np.sqrt(var), vl))
                h.append(plt.scatter(xhat_dict[cls][:, 0], xhat_dict[cls]
                                     [:, 1], s=70, c=colors[i], alpha=0.5, marker='x'))
                legenda.append('fault type - ' + str(cls))
                i = i + 1

            ##############
            # Plotting t2 ellipse
            _, _, t2 = self.calculate_t2_statistics(k, plot=False)
            x, y = np.meshgrid(np.linspace(-10,10,1000), np.linspace(-10, 10, 1000))
            right_side = t2['thr']
            left_side = np.power(
                x, 2) / t2['lamb'][0, 0] + np.power(y, 2) / t2['lamb'][1, 1]

            plt.contour(x, y, right_side - left_side, [0])
            
            ##############
            plt.legend(h, legenda)
            plt.grid(True)
            plt.title('Principal components')
            plt.xlabel('First principal component')
            plt.ylabel('Second principal component')
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plt.hold(True)
            h = []
            h_legend = []
            legenda = []
            for cls in x_dict.keys():
                x_cls = x_dict[cls]
                xhat_dict[cls] = np.real(
                    np.dot((x_cls - mu) / np.sqrt(var), vl))
                h.append(ax.scatter(xhat_dict[cls][:, 0], xhat_dict[cls][
                         :, 1], xhat_dict[cls][:, 2], s=70, c=colors[i], alpha=0.5, marker='x'))
                legenda.append('fault type - ' + str(cls))
                h_legend.append(lines.Line2D(
                    [0], [0], linestyle="none", c=colors[i], marker='x'))
                i = i + 1

            ax.legend(h_legend, legenda, numpoints=1)

            # ax.legend(h, legenda)
            ax.grid(True)
            plt.title('Principal components')
            ax.set_xlabel('First principal component')
            ax.set_ylabel('Second principal component')
            ax.set_zlabel('Third principal component')
        plt.savefig(self.path + 'figures/' + name + '_k=' + str(k) + '.eps')
        plt.clf()

    def calculate_t2_statistics(self, k, name='t2', plot=True):
        from scipy.stats import f

        index = np.nonzero(self.y == 0)
        x_normal = self.x[index[0], :]

        xhat_normal, wl, vl = pca(x_normal, k)  # dados na nova base

        mu = np.mean(x_normal, 0)

        n = x_normal.shape[0]
        p = 0.05
        dfn = k
        dfd = n - k
        f_density = f.ppf(1.0 - p / 2, dfn, dfd)
        t2_thr = (1.0 * k * (n - 1) / (n - k)) * f_density

        # dados na nova base utilizndo o pca calculado apenas com dados
        # normais.

        xhat = np.real(np.dot(self.x - mu, vl))

        lamb = np.eye(k)
        for i in range(0, k):
            lamb[i, :] = wl[i] * lamb[i,:]

        t2_statistics = np.array([])
        for i in range(0, xhat.shape[0]):
            t2_statistics = np.append(t2_statistics, np.dot(
                np.dot(xhat[i, :], np.linalg.inv(lamb)), np.transpose(xhat[i,:])))

        t2_statistics = np.reshape(t2_statistics, (-1, 1))
        yhat = t2_statistics > t2_thr

        time_fault = np.nonzero(self.y != 0)[0][0]

        if plot:
            plt.semilogy(np.arange(0, xhat.shape[0]), t2_statistics)
            plt.hold(True)
            plt.semilogy(t2_thr * np.ones_like(t2_statistics), '--g')
            plt.axvline(x=time_fault, ymin=0, ymax=1, linestyle='--', color='r')
    
            plt.xlabel('time (min)')
            plt.ylabel('log($t^2$ statistics)')
            plt.legend(
                ['$log(t^2)$ statistic', '$t^2$ threshold', 'fault starting time'])
    
            plt.title(
                '$t^2$ statistic over time for fault {:s} using {:d} principal components'.format(name, k))
            plt.savefig(
                self.path + 'figures/' + name + 't2vstime_k=' + str(k) + '.eps')
            plt.clf()
        return yhat, t2_statistics, {'lamb': lamb, 'thr': t2_thr}


def pca(x, k=0):
    '''
    Analise de componentes principais da matriz de dados X.
    Entrada:
        x:    matriz N x M em que cada linha representa
    uma amostra com M atributos.
        k:    numero de componentes principais a serem utilizadas.
    Retorna:
        xhat: retorna os dados na nova base utilizando K componentes principais.
        wl:   k autovalores.
        vl:   k autovetores que transformam os dados em x para xhat.
    '''

    n = x.shape[0]  # numero de amostras
    if k == 0:
        k = n
    elif k > n:
        k = n

    # calcula a matriz de correlacao
    c = np.cov(np.transpose(x))
    # c2 = correlacao(x)

    # calcula os autovalores / autovetores da matriz acima
    w, v = np.linalg.eig(c)

    index = np.argsort(w)  # coloca os autovalores de forma crescente
    aux = [i for i in index[-1::-1]]
    index = aux  # agora de forma descrescente
    w = w[index]
    v = v[:, index]

    wl = w[:k]
    vl = v[:, :k]

    mu = np.mean(x, 0)
    var = np.var(x, 0)

    xhat = np.real(np.dot((x - mu), vl))  # dados na nova base

    return xhat, wl, vl


def correlacao(x):
    '''
    Calcula a matriz de correlacao dos dados em x.
    Entrada:
        x:    matriz N x M em que cada linha representa
    uma amostra com M atributos.
    Saida:
        matriz de autocorrelacao de x.
    '''
    n = x.shape[0]  # numero de amostras

    mu = np.mean(x, 0)
    var = np.var(x, 0)
    # Calcular a matriz C de autocovariancia/correlacao dos dados;

    # Normalizacao
    # subtrai a media e divide pelo desvio padrao
    x_hat = (x - mu) / np.sqrt(var)

    # calcula a matriz de correlacao
    c = np.dot(np.transpose(x_hat), x_hat) / (n - 1)

    return c


def accuracy_fault_no_fault(y, yhat):
    '''
    Calcula a acuracia entre os vetores.
    Entradas:
        y:    vetor de classes reais.
        yhat: vetor de classes estimadas.
    Saida:
        acc: acuracia.
    '''
    n = len(yhat)
    y = np.reshape(np.array(y), (-1, 1))
    yhat = np.reshape(np.array(yhat), (-1, 1))

    index = np.where(yhat == True)[0]
    yhat[index] = 1
    index = np.where(yhat == False)[0]
    yhat[index] = 0

    index = np.where(y != 0)[0]
    y[index] = 1

    hit = np.where(y == yhat)

    acc = 1.0 * hit[0].size / n

    return acc
