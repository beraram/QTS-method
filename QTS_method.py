'''
Bernardo de Azevedo Ramos 27/04/2023

Unsupervised Drift Detection Method - QTS

'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from skmultiflow.data.data_stream import DataStream
from skmultiflow.data import SEAGenerator
from skmultiflow.data.file_stream import FileStream
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.data.stagger_generator import STAGGERGenerator
from skmultiflow.data.led_generator_drift import LEDGeneratorDrift
from skmultiflow.data.mixed_generator import MIXEDGenerator
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection import DDM
from skmultiflow.drift_detection.eddm import EDDM
# from my_eddm import EDDM  # Este aqui é usado apenas para bases reais
from skmultiflow.drift_detection import PageHinkley
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from skmultiflow.bayes import NaiveBayes
from skmultiflow.lazy import KNN
import os
import psutil
import time
from QTree_plus import Node as NDT
# from NovaQTreeHeight import Node as NDTheight
from sklearn import metrics

# Plot Model Classes
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# Dinamic height
from scipy.stats import kde  # Densidade
import math
import random

# Derivada
from numpy import diff

# Save results
import pickle

# live plots
import matplotlib.animation as animation
from matplotlib import style


#Define QuadTree
def catchallNEW(tree, level=0):
    catchdata = []
    if tree.data is None:
        level = level + 1
        if tree.filhos is not None:
            for i, j in enumerate(tree.filhos):
                catchdata += catchallNEW(tree.filhos[j], level)  # Todos os pontos
    else:
        # print(tree.data, level)
        datafolha = np.mean(tree.data, axis=0)
        datafolha = datafolha.tolist()
        datafolha = [datafolha]
        # print(datafolha, level, 'NEW')
        return datafolha  # Retorna o ponto medio das folhas
        # return tree.data  # Retorna os pontos de cada folha

    return catchdata


# Definição das iterações desejadas
switch = ['Iteração_01']

for ind, auto_h in enumerate(switch):

    # Dataset reading
    df = pd.read_csv("D:/Users/ricar/Desktop/Mestrado/Dissertação de Mestrado/Bases/Toy.csv")

    # Sets all the variables to a scale from 0 to 1
    df = df.astype(float)
    scaler = MinMaxScaler()
    df.iloc[:, 0:df.shape[1] - 1] = scaler.fit_transform(df.iloc[:, 0:df.shape[1] - 1])
    stream = DataStream(df)

    # Memory and processor time statistics
    pid = os.getpid()
    ps = psutil.Process(pid)
    start = time.time()

    ''' SETTING THE INPUT VARIABLES '''

    # Create a QuadTree
    height = 4 # Quadtree height

    X_train, y_train = stream.next_sample()  # Takes data from the stream to set the dimensionality of the quadtree
    mid = [0.5] * X_train[0].size  # Self adapt to first data dimension
    ndta = NDT(mid, 0.5, height)
    n_samples = 1  # Counting first sample used to create the tree

    # Drift positions
    drift = []  # Position in which the drifts are located at the dataset
    drift_found = []  # Position in which the drifts were detected

    # Amount of data (occupancy) and derivative window
    data_QT = [0, 1]  # Amount of data in summarized QTS
    derivative_w = []  # Window with the values of the first derivative of sw
    derivative_rd = 0  # Value of the average derivative of recent data from the derivatives window f'(sw)

    derivate_b = []  # list of values of the average derivative of recent data from the derivative window

    # QTS sliding window and QTS sliding window size
    QTS_win = []  # List of data to be inserted in QTS
    QTS_win_size = 500  # Size of the sliding window of data to be inserted into the QTS

    # Flags
    flag_dv_change = 'false'  # Signals presence/status of drift when detected with derivative mean change detection
    flag_initial = 'true'  # Indicates that the data flow is starting, first window being mounted
    flag_ocup_change = 'false' # Signals presence/status of drift when detected with occupancy baseline change detection
    repete_flag = 'false' # Flag that determines whether samples in a drift state will be counted.
   
    # Initialization of other variables
    threshold_dv = 0 # Threshold value in derivative mean change detection
    cont_drift = 0 # Auxiliary counter
    ocup_ref = 0 # Reference occupancy value that will be compared with the mean value of the sliding window
    ocup_win = 0 # Occupancy value of the sliding window
    ocup_dif = 0 # Exponential moving average of the occupancy sliding window
    std_ref = 0 # Standard deviation of the occupancy reference window
    std_win = 0 # Standard deviation of the occupancy sliding window
    cont_ocup_drift = 0 # Auxiliary counter
    ocup_win_nmean = 0 # Occupancy normal mean value of the sliding window
    rho = 0.1 # percentage of recent data in derivative sliding window
    win_tam_size = 1500 # Summarized sliding window size
    alpha = 1 # Multiplicative threshold
    beta = 3 # Multiplicative threshold

    '''STREAMING DATA'''

    # Run test-then-train loop for max_samples or while there is data in the stream
    while stream.has_more_samples():  # 59800
        n_samples += 1
        X, y = stream.next_sample()

        # Running process
        print('.', end="")
        if not n_samples % 100:
            print(end="\r")
            print('Streaming :', n_samples, ' ', end="")

        ### drift detectors ###
        ### QTS_NS


        # Sliding window with the samples to be inserted into the QTS
        QTS_win.append(X[0])
        if len(QTS_win) > QTS_win_size:
            QTS_win = QTS_win[1:]  # All but the first, i.e. delete the oldest data

        #### Saturation Tree ####
        # Todos os dados do fluxo são inseridos na árvore.
        ndta = NDT(mid, 0.5, height)
        for i, j in enumerate(QTS_win):
            ndta.insert(j)

        # Occupancy (amount of data of Quadtree)
        data_QT.append(ndta.number_leaf())

        ## Coding the growth of the amount of data in the tree (first derivative)
        if data_QT[-1] == data_QT[-2]:
            derivative_w.append(0)

        if data_QT[-1] > data_QT[-2]:
            derivative_w.append(1)

        if data_QT[-1] < data_QT[-2]:
            derivative_w.append(-1)

        # Define recent data size of the derivative sliding window
        attenuator = 1  # Changing the intensity of the derivative
        rd = int(rho * QTS_win_size)


        # Setting the initial reference window
        if (len(data_QT) > (QTS_win_size + win_tam_size)) and (flag_initial == 'true'):
            ocup_ref = np.mean(data_QT[-win_tam_size:])
            std_ref = np.std(data_QT[-win_tam_size:])
            flag_initial = 'false'

        # Extracting the characteristics of the sliding window
        if (len(data_QT) > (QTS_win_size + win_tam_size)):
            ocup_df = pd.DataFrame(data_QT[-win_tam_size:])
            ocup_win_df = ocup_df.ewm(span = win_tam_size).mean()
            ocup_win = ocup_win_df.iloc[win_tam_size-1,0]
            ocup_win_nmean = np.mean(data_QT[-win_tam_size:])
            std_win = np.std(data_QT[-win_tam_size:])
            #ocup_win = np.mean(data_QT[-win_tam_size:])

        # Setting derivative recent data and derivative mean change detection threshold
        if len(derivative_w) > win_tam_size:
            #derivative_rd = (np.mean(derivative_w[-win_tam_size:]) * attenuator)
            threshold_dv = (np.mean(derivative_w[-win_tam_size:-(rd-1)]) + alpha * np.std(derivative_w[-win_tam_size:-(rd-1)]))
            derivative_rd = (np.mean(derivative_w[-rd:]) * attenuator)
            derivate_b.append(derivative_rd)

        # Derivative mean change detection
        if (repete_flag == 'true'):
            if (abs(derivative_rd) > threshold_dv):
                drift_found.append(n_samples)



        if (repete_flag == 'false'):
            if (abs(derivative_rd) > threshold_dv) and (flag_dv_change == 'false'):
                drift_found.append(n_samples)
                flag_dv_change = 'true'

            # Wait for the new concept re-occupancy of sliding windows
            if (flag_dv_change == 'true'):
                if cont_drift < (QTS_win_size + win_tam_size):
                    cont_drift += 1
                else:
                    print(cont_drift)
                    cont_drift = 0
                    flag_dv_change = 'false'
                    ocup_ref = ocup_win_nmean
                    std_ref = std_win
                    print(n_samples)

        # Occupancy baseline change detection
        if (flag_initial == 'false'):
            #ocup_dif = abs((ocup_ref - ocup_win) / ocup_ref)
            ocup_dif = ocup_win

        if ((ocup_dif > (ocup_ref + beta*std_ref)) or (ocup_dif < (ocup_ref - beta*std_ref))) and (flag_dv_change == 'false') and (flag_ocup_change == 'false'):
            drift_found.append(n_samples)
            flag_ocup_change = 'true'

        # Wait for the new concept re-occupancy of sliding windows
        if flag_ocup_change == 'true':
            if cont_ocup_drift < win_tam_size:
                cont_ocup_drift += 1
            else:
                cont_ocup_drift = 0
                ocup_ref = ocup_win_nmean
                std_ref = std_win
                flag_ocup_change = 'false'


        ### END QTS_method

    ## Results

    print(' ')
    print('{} samples analyzed.'.format(n_samples))
    print('Drift positions:', drift)
    print('Positions where drift were detected:', drift_found)
    print('Amount of drifts:', len(drift_found))

    # time and memory spend
    memoryUse = ps.memory_info()
    print('Memory used: ', memoryUse.rss / 1000000, "MB usados no PID")
    end = time.time()
    print('Time spent: ', end - start)
    print('Quadtree height: ', height)

    ## Plot results

    plt.figure(1)
    xline_l = list(range(len(data_QT)))
    plt.plot(xline_l, data_QT, 'r', label='Occupancy')
    for i, j in enumerate(drift_found):  # Plotar a posição real dos drifts
        plt.axvline(j, color='r', linestyle=':')  # Plotar a posição real do drift
    # plt.ylim(ymin=0.5, ymax=1.0)  # this line
    #for i, k in enumerate(drift_found_stu):  # Plotar a posição real dos drifts
        #drifts_studd = plt.axvline(k, color='g', linestyle=':', label='Drift STUDD')
    #for i, l in enumerate(drift_found_d3):  # Plotar a posição real dos drifts
        #drifts_d3 = plt.axvline(l, color='b', linestyle=':', label='Drift D3')
    #for i, m in enumerate(drift_found_ocdd):  # Plotar a posição real dos drifts
        #drifts_ocdd = plt.axvline(m, color='y', linestyle=':', label='Drift OCDD')
    plt.legend(['Occupancy', 'Drifts'], loc='lower right')  # loc='upper lower right'
    plt.grid(True)
    plt.show()

    # Plot derivatives
    plt.figure(2)
    xline_l = list(range(win_tam_size, len(derivate_b) + win_tam_size))
    plt.plot(xline_l, derivate_b, 'b'),
    plt.legend(['Derivative'], loc='upper right')  # loc='upper lower right'
    plt.grid(True)
    for i, j in enumerate(drift_found):  # Plotar a posição real dos drifts
        plt.axvline(j, color='g', linestyle=':')  # Plotar a posição real do drift
    plt.show()


