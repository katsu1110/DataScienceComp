import numpy as np
import pandas as pd

# keras
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, LambdaCallback
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.layers import Input, Layer, Dense, Concatenate, Reshape, Dropout, merge, Add, BatchNormalization, GaussianNoise
from keras.layers.embeddings import Embedding
from keras import backend as K
from keras.layers import Layer
from keras.callbacks import *
import tensorflow as tf
import math

# # visualize
# import matplotlib.pyplot as plt
# import matplotlib.style as style
# import seaborn as sns
# from matplotlib import pyplot
# from matplotlib.ticker import ScalarFormatter
# sns.set_context("talk")
# style.use('fivethirtyeight')

# utils
import os, sys
mypath = os.getcwd()
sys.path.append(mypath + '/code/')
from nn_utils import Mish, LayerNormalization, CyclicLR
from permutation_importance import PermulationImportance
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from base_models import BaseModel

class NeuralNetworkModel(BaseModel):
    """
    MLP wrapper: for now not so flexible

    """

    def train_model(self, train_set, val_set):
        # MLP model
        inputs = []
        n_neuron = self.params['hidden_units']
        if len(self.categoricals) > 0:
            embeddings = []
            embedding_out_dim = self.params['embedding_out_dim']
            for i in self.categoricals:
                input_ = Input(shape=(1,))
                embedding = Embedding(int(np.absolute(self.train_df[i]).max() + 1), embedding_out_dim, input_length=1)(input_)
                embedding = Reshape(target_shape=(embedding_out_dim,))(embedding)
                inputs.append(input_)
                embeddings.append(embedding)
            input_numeric = Input(shape=(len(self.features) - len(self.categoricals),))
            embedding_numeric = Dense(n_neuron)(input_numeric)
            embedding_numeric = Mish()(embedding_numeric)
            inputs.append(input_numeric)
            embeddings.append(embedding_numeric)
            x = Concatenate()(embeddings)
        else: # no categorical features
            inputs = Input(shape=(len(self.features), ))
            x = Dense(n_neuron)(inputs)
            x = Mish()(x)
            x = Dropout(self.params['hidden_dropout'])(x)
            x = LayerNormalization()(x)
            
        # more layers
        for i in np.arange(self.params['hidden_layers'] - 1):
            x = Dense(n_neuron // (2 * (i+1)))(x)
            x = Mish()(x)
            x = Dropout(self.params['hidden_dropout'])(x)
            x = LayerNormalization()(x)
        
        # output
        if self.task == "regression":
            out = Dense(1, activation="linear", name = "out")(x)
            loss = "mse"
        elif self.task == "binary":
            out = Dense(1, activation='sigmoid', name = 'out')(x)
            loss = "binary_crossentropy"
        elif self.task == "multiclass":
            out = Dense(train_set['y'].nunique(), activation='softmax', name = 'out')(x)
            loss = "categorical_crossentropy"
        model = Model(inputs=inputs, outputs=out)

        # compile
        if self.params['optimizer']['type'] == 'adam':
            model.compile(loss=loss, optimizer=Adam(lr=self.params['optimizer']['lr'], beta_1=0.9, beta_2=0.999, decay=1e-04))
        elif self.params['optimizer']['type'] == 'sgd':
            model.compile(loss=loss, optimizer=SGD(lr=self.params['optimizer']['lr'], decay=1e-6, momentum=0.9))

        # callbacks
        er = EarlyStopping(patience=10, min_delta=1e-4, restore_best_weights=True, monitor='val_loss')
        ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
        history = model.fit(train_set['X'], train_set['y'], callbacks=[er, ReduceLR],
                            epochs=self.params['epochs'], batch_size=self.params['batch_size'],
                            validation_data=[val_set['X'], val_set['y']])

        # permutation importance to get a feature importance (off in default)
        # fi = PermulationImportance(model, train_set['X'], train_set['y'], self.features)
        fi = np.zeros(len(self.features)) # no feature importance computed
        return model, fi

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = {'X': x_train, 'y': y_train}
        val_set = {'X': x_val, 'y': y_val}
        return train_set, val_set

    # def plot_loss(self):
    #     # Plot training & validation loss values
    #     plt.plot(history.history['loss'])
    #     plt.plot(history.history['val_loss'])
    #     plt.title('Model loss')
    #     plt.ylabel('Loss')
    #     plt.xlabel('Epoch')
    #     plt.legend(['Train', 'Test'], loc='upper left ')
    #     plt.show()

    def get_params(self):
        """
        adapted from https://github.com/ghmagazine/kagglebook/blob/master/ch06/ch06-03-hopt_nn.py
        """
        params = {
            'input_dropout': 0.0,
            'hidden_layers': 2,
            'hidden_units': 128,
            'embedding_out_dim': 4,
            # 'hidden_activation': 'relu', # use always mish
            'hidden_dropout': 0.05,
            # 'batch_norm': 'before_act', # use always LayerNormalization
            'optimizer': {'type': 'adam', 'lr': 1e-4},
            'batch_size': 256,
            'epochs': 80
        }

        return params