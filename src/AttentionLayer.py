#__*__encoding=utf-8__*__
import os, sys, time, datetime
import pandas as pd
import numpy as np
import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import ModelCheckpoint
from keras import Input, Model, losses, Sequential
from keras.activations import relu, sigmoid
from keras.layers import Dense, Bidirectional, LSTM, Dropout, Activation, Flatten, concatenate, Reshape
from keras.optimizers import Adam, SGD
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import xgboost as xgb
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC

class AttentionLayer(Layer):
    def __init__(self, attention_size=None, **kwargs):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size

        self.W = self.add_weight(name='att_weight', shape=(hidden_size, self.attention_size),
                                initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,),
                                initializer='uniform', trainable=True)
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,),
                                initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        self.V = K.reshape(self.V, (-1, 1))
        H = K.tanh(K.dot(inputs, self.W) + self.b)
        score = K.softmax(K.dot(H, self.V), axis=1)
        outputs = K.sum(score * inputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

