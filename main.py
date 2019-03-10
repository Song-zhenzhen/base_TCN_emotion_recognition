from utils import data_generator
from sklearn.model_selection import StratifiedKFold, train_test_split
from tcn import compiled_tcn
import scipy.io
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import os
import time
#import model_ensemble
from keras.callbacks import TensorBoard
import glob
from keras.layers import *
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import np_utils
from snapshot import Snapshot
import model_ensemble
skf = StratifiedKFold(n_splits = 5, random_state = 0, shuffle = True)

for k in range(2):
    n=k+17
    print(n)
    data=scipy.io.loadmat('all/df_all%d.mat'%(n))['data']
    data = data.reshape(2400,40,16)
    data3=np.zeros(640)
    for i in range(2400):
        data2=np.zeros(1)
        for j in range(40):
            data1=data[i][j]
            data2=np.hstack((data2,data1))
        data2=data2[1:]
        data3=np.vstack((data3,data2))
    data3 = data3[1:]
    data=data3.reshape(2400,160,4)
    #data=data.reshape(2400,160,4)
    #df1=np.ones(320)
    #for j in range(2520):
     #   df=data[j][0:320]
    #    df1=np.vstack((df1,df))
   # data=df1[1:2521].reshape(2520,40,8)
    label=scipy.io.loadmat('label_4_kmeans/l%d.mat'%(n))['labels']
    label=label.reshape(-1)
    #label2=np.zeros(1)
    #for i in range(40):
        #label1=label[(63*i+3):(63*(i+1))]
        #label2 = np.hstack((label2,label1))
    #label=label2[1:]
    #for index_train, index_test in skf.split(data, label):
    index=np.ones(2400)
    for i in range(2400):
        index[i]=i
    np.random.shuffle(index)
    index=index.astype('int64')
    index_train=index[0:1900]
    index_test=index[1900:2400]
        
    x_total = data[index_train]
    x_test = data[index_test]
    y_total = label[index_train]
    y_test = label[index_test]
    x_train, x_valid, y_train, y_valid = train_test_split(x_total, y_total, 
                                                    test_size = 0.1, random_state = 1)
    y_valid1 = np.expand_dims(y_valid, axis = 1)
    #y_test = np.expand_dims(y_test, axis = 1)
    y_valid3 = to_categorical(y_valid, num_classes = 4)
    model = compiled_tcn(return_sequences=False,
                         num_feat=4,
                         num_classes=4,
                         nb_filters=50                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ,
                         kernel_size=4,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=2,
                         max_len=x_train[0:1].shape[1],
                         activation='norm_relu',
                         
                         use_skip_connections=True)
       
    score1=np.ones(2)
    #model.summary()
    cbs=[Snapshot('snapshots', nb_epochs=120, verbose=0, nb_cycles=8)]
    #s_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    #logs_path='G:/deap/Result/logs/log_%s'%(s_time)
    #try:
    #    os.makedirs(logs_path)
    #except:
    #    pass
    #tensorboard=TensorBoard(log_dir=logs_path,histogram_freq=1,write_graph=True)
    for epoch in range(20):
        print('epoch:',epoch+1)
        model.fit(x_train, y_train, batch_size=20, epochs=2,verbose=0,
                  validation_data=(x_valid,y_valid))
        #model.fit(x_train, y_train, batch_size=10, epochs=121,verbose=0,
                  #validation_data=(x_valid,y_valid),callbacks=cbs)
        score = model.evaluate(x_test, y_test, batch_size=10)
        score1=np.vstack((score1,score))
        print(score)
        #print(s)
    #scipy.io.savemat('G:/deap/Result/d%d.mat'%(k+1),{'data':score1})
        #model.fit(x_train, y_train.squeeze().argmax(axis=1), epochs=100,
                  #validation_split=1/3)
#data=scipy.io.loadmat('G:/deap/Result/d18.mat')['data']
#print(data)
'''
    model1 = compiled_tcn(return_sequences=False,
                         num_feat=2,
                         num_classes=4,
                         nb_filters=50                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ,
                         kernel_size=4,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=2,
                         max_len=x_train[0:1].shape[1],
                         activation='norm_relu',
                         use_skip_connections=True)
    model1.load_weights('snapshots/weights_cycle_0.h5')
    model2 = compiled_tcn(return_sequences=False,
                         num_feat=2,
                         num_classes=4,
                         nb_filters=50                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ,
                         kernel_size=4,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=2,

                         max_len=x_train[0:1].shape[1],
                         activation='norm_relu',
                         use_skip_connections=True)
    model2.load_weights('snapshots/weights_cycle_1.h5')

    model3 = compiled_tcn(return_sequences=False,
                             num_feat=2,
                             num_classes=4,
                             nb_filters=50                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ,
                             kernel_size=4,
                             dilations=[2 ** i for i in range(9)],
                             nb_stacks=2,
                             max_len=x_train[0:1].shape[1],
                             activation='norm_relu',
                             use_skip_connections=True)
    model3.load_weights('snapshots/weights_cycle_2.h5')
    
    model4 = compiled_tcn(return_sequences=False,
                             num_feat=2,
                             num_classes=4,
                             nb_filters=50                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ,
                             kernel_size=4,
                             dilations=[2 ** i for i in range(9)],
                             nb_stacks=2,
                             max_len=x_train[0:1].shape[1],
                             activation='norm_relu',
                             use_skip_connections=True)
    model4.load_weights('snapshots/weights_cycle_3.h5')
    
    model5 = compiled_tcn(return_sequences=False,
                             num_feat=2,
                             num_classes=4,
                             nb_filters=50                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ,
                             kernel_size=4,
                             dilations=[2 ** i for i in range(9)],
                             nb_stacks=2,
                             max_len=x_train[0:1].shape[1],
                             activation='norm_relu',
                             use_skip_connections=True)
    model5.load_weights('snapshots/weights_cycle_4.h5')

    model6 = compiled_tcn(return_sequences=False,
                         num_feat=2,
                         num_classes=4,
                         nb_filters=50                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ,
                         kernel_size=4,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=2,
                         max_len=x_train[0:1].shape[1],
                         activation='norm_relu',
                         use_skip_connections=True)
    model6.load_weights('snapshots/weights_cycle_5.h5')

    model7 = compiled_tcn(return_sequences=False,
                         num_feat=2,
                         num_classes=4,
                         nb_filters=50                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ,
                         kernel_size=4,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=2,
                         max_len=x_train[0:1].shape[1],
                         activation='norm_relu',
                         use_skip_connections=True)
    model7.load_weights('snapshots/weights_cycle_6.h5')

    model8 = compiled_tcn(return_sequences=False,
                         num_feat=2,
                         num_classes=4,
                         nb_filters=50                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ,
                         kernel_size=4,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=2,
                         max_len=x_train[0:1].shape[1],
                         activation='norm_relu',
                         use_skip_connections=True)
    model8.load_weights('snapshots/weights_cycle_7.h5')
    
    

    acc1 = model1.evaluate(x_test, y_test, batch_size=10)
    acc2 = model2.evaluate(x_test, y_test, batch_size=10)
    acc3 = model3.evaluate(x_test, y_test, batch_size=10)
    acc4 = model4.evaluate(x_test, y_test, batch_size=10)
    acc5 = model5.evaluate(x_test, y_test, batch_size=10)
    acc6 = model6.evaluate(x_test, y_test, batch_size=10)
    acc7 = model7.evaluate(x_test, y_test, batch_size=10)
    acc8 = model8.evaluate(x_test, y_test, batch_size=10)
    
        
    print('模型1精度:', acc1)
    print('模型2精度:', acc2)
    print('模型3精度:', acc3)
    print('模型4精度:', acc4)
    print('模型5精度:', acc5)
    print('模型6精度:', acc6)
    print('模型7精度:', acc7)
    print('模型8精度:', acc8)


    models=[model1,model2,model3,model4,model5,model6,model7,model8]
    y_test=np.expand_dims(y_test,axis=1)
    y_pre1,ensemble_acc1 = model_ensemble.vote_ensemble(x_test, y_test, models)
    print('集成模型1精度:', ensemble_acc1)
'''








