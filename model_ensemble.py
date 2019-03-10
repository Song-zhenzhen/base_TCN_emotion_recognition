# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 21:48:34 2018

@author: 振振
"""
from __future__ import print_function
from keras.optimizers import SGD
import numpy as np
import pandas as pd
import scipy.io
from sklearn import svm
from keras.optimizers import Adam
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Bidirectional
from keras.layers import SimpleRNN,LSTM,Dropout
from keras.optimizers import RMSprop
from keras.layers.recurrent import GRU
from keras import regularizers
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras import initializers, constraints
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import keras.backend.tensorflow_backend as KTF
from keras.initializers import glorot_uniform
from keras import layers
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,ZeroPadding2D,add,Activation
from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation,Input,BatchNormalization,AveragePooling2D,concatenate
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping


    
def evaluate_acc(x_test, y_test, model):
    pred = model.predict(x_test)
    pred = np.argmax(pred, axis = 1)
    pred = np.expand_dims(pred, axis = 1)
    error = np.sum(np.not_equal(pred, y_test)) / y_test.shape[0]
    return 1-error
def vote_ensemble(x_test, y_test, models):    
    out0 = to_categorical(np.argmax(models[0].predict(x_test), axis = 1), num_classes = 4)
    out1 = to_categorical(np.argmax(models[1].predict(x_test), axis = 1), num_classes = 4)
    out2 = to_categorical(np.argmax(models[2].predict(x_test), axis = 1), num_classes = 4)
    out3 = to_categorical(np.argmax(models[3].predict(x_test), axis = 1), num_classes = 4)
    out4 = to_categorical(np.argmax(models[4].predict(x_test), axis = 1), num_classes = 4)
    out5 = to_categorical(np.argmax(models[5].predict(x_test), axis = 1), num_classes = 4)
    out6 = to_categorical(np.argmax(models[6].predict(x_test), axis = 1), num_classes = 4)
    out7 = to_categorical(np.argmax(models[7].predict(x_test), axis = 1), num_classes = 4)
    
    

    out = out0+out1+out2+out3+out4+out5+out6+out7
    y_pre = np.argmax(out, axis = 1)
    y_pre = np.expand_dims(y_pre, axis = 1)
    error = np.sum(np.not_equal(y_pre, y_test)) / y_test.shape[0]
    return y_pre,1-error

#模型集成2：自适应加权
def weight_ensemble(x_test, y_test, models, w):
    out0 = models[0].predict(x_test)
    out1 = models[1].predict(x_test)
    out2 = models[2].predict(x_test)
    out3 = models[3].predict(x_test)
    out4 = models[4].predict(x_test)

    out = (w[0]*out0+w[1]*out1+w[2]*out2+w[3]*out3+w[4]*out4)/sum(w)
    y_pre = np.argmax(out, axis = 1)
    y_pre = np.expand_dims(y_pre, axis = 1)
    error = np.sum(np.not_equal(y_pre, y_test)) / y_test.shape[0]
    return y_pre,1-error
'''
model1=Sequential()
model1.add(keras.layers.core.Masking(mask_value=0., input_shape=(40,16)))
model1.add(GRU(100, return_sequences=False, input_shape=(40,16)))
model1.add(Dropout(0.5))
model1.add(Dense(2, activation='sigmoid'))

model2=Sequential()
model2.add(keras.layers.core.Masking(mask_value=0., input_shape=(40,16)))
model2.add(GRU(100, return_sequences=False, input_shape=(40,16)))
model2.add(Dropout(0.5))
model2.add(Dense(2, activation='sigmoid'))

model3=Sequential()
model3.add(keras.layers.core.Masking(mask_value=0., input_shape=(40,16)))
model3.add(GRU(100, return_sequences=False, input_shape=(40,16)))
model3.add(Dropout(0.5))
model3.add(Dense(2, activation='sigmoid'))

model4=Sequential()
model4.add(keras.layers.core.Masking(mask_value=0., input_shape=(40,16)))
model4.add(GRU(100, return_sequences=False, input_shape=(40,16)))
model4.add(Dropout(0.5))
model4.add(Dense(2, activation='sigmoid'))

model5=Sequential()
model5.add(keras.layers.core.Masking(mask_value=0., input_shape=(40,16)))
model5.add(GRU(100, return_sequences=False, input_shape=(40,16)))
model5.add(Dropout(0.5))
model5.add(Dense(2, activation='sigmoid'))


model1.load_weights('snapshots/weights_cycle_0.h5')
model2.load_weights('snapshots/weights_cycle_1.h5')
model3.load_weights('snapshots/weights_cycle_2.h5')
model4.load_weights('snapshots/weights_cycle_3.h5')
model5.load_weights('snapshots/weights_cycle_4.h5')

model1.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=0.1, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
model2.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=0.1, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
model3.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=0.1, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
model4.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=0.1, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
model5.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=0.1, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

score1 = model1.evaluate(x_valid, y_valid3, batch_size=10)
score2 = model2.evaluate(x_valid, y_valid3, batch_size=10)
score3 = model3.evaluate(x_valid, y_valid3, batch_size=10)
score4 = model4.evaluate(x_valid, y_valid3, batch_size=10)
score5 = model5.evaluate(x_valid, y_valid3, batch_size=10)

score = [score1, score2, score3, score4, score5]
b = sorted(score)
w1 = b.index(score[0])
w2 = b.index(score[1])
w3 = b.index(score[2])
w4 = b.index(score[3])
w5 = b.index(score[4])

w0 = [w1,w2,w3,w4,w5]
w = [j+1 for j in w0]
np.save('G:/dataset/emotion/gan_data/snapshot/weights/w',w)

w_raw = np.load('G:/dataset/emotion/gan_data/snapshot/weights/w.npy')
w = w_raw.tolist()

acc1 = model1.evaluate(x_test, y_test, batch_size=10)
acc2 = model2.evaluate(x_test, y_test, batch_size=10)
acc3 = model3.evaluate(x_test, y_test, batch_size=10)
acc4 = model4.evaluate(x_test, y_test, batch_size=10)
acc5 = model5.evaluate(x_test, y_test, batch_size=10)

print('模型1精度:', acc1)
print('模型2精度:', acc2)
print('模型3精度:', acc3)
print('模型4精度:', acc4)
print('模型5精度:', acc5)

models=[model1,model2,model3,model4,model5]
y_test=np.argmax(y_test,axis=1)
y_test=np.expand_dims(y_test,axis=1)

y_pre1,ensemble_acc1 = vote_ensemble(x_test, y_test, models)
print('集成模型1精度:', ensemble_acc1)

y_pre2,ensemble_acc2 = weight_ensemble(x_test, y_test, models, w)
print('集成模型2精度:',ensemble_acc2)'''














