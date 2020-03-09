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
import tensorflow as tf
import math

# helper
import os, sys
mypath = os.getcwd()
sys.path.append(mypath + '/code/')
from nn_utils import Mish, LayerNormalization, CyclicLR

def nn_model(cls, train_set, val_set):
    """
    NN hyperparameters and models
    """

    # adapted from https://github.com/ghmagazine/kagglebook/blob/master/ch06/ch06-03-hopt_nn.py
    params = {
        'input_dropout': 0.0,
        'hidden_layers': 2,
        'hidden_units': 128,
        'embedding_out_dim': 4,
        # 'hidden_activation': 'relu', # use always mish
        'hidden_dropout': 0.08,
        # 'batch_norm': 'before_act', # use always LayerNormalization
        'optimizer': {'type': 'adam', 'lr': 1e-4},
        'batch_size': 128,
        'epochs': 80
    }

    # NN model architecture
    inputs = []
    n_neuron = params['hidden_units']

    # embedding for categorical features 
    if len(cls.categoricals) > 0:
        embeddings = []
        embedding_out_dim = params['embedding_out_dim']
        for i in cls.categoricals:
            input_ = Input(shape=(1,))
            embedding = Embedding(int(np.absolute(cls.train_df[i]).max() + 1), embedding_out_dim, input_length=1)(input_)
            embedding = Reshape(target_shape=(embedding_out_dim,))(embedding)
            inputs.append(input_)
            embeddings.append(embedding)
        input_numeric = Input(shape=(len(cls.features) - len(cls.categoricals),))
        embedding_numeric = Dense(n_neuron)(input_numeric)
        embedding_numeric = Mish()(embedding_numeric)
        inputs.append(input_numeric)
        embeddings.append(embedding_numeric)
        x = Concatenate()(embeddings)

    else: # no categorical features
        inputs = Input(shape=(len(cls.features), ))
        x = Dense(n_neuron)(inputs)
        x = Mish()(x)
        x = Dropout(params['hidden_dropout'])(x)
        x = LayerNormalization()(x)
        
    # more layers
    for i in np.arange(params['hidden_layers'] - 1):
        x = Dense(n_neuron // (2 * (i+1)))(x)
        x = Mish()(x)
        x = Dropout(params['hidden_dropout'])(x)
        x = LayerNormalization()(x)
    
    # output
    if cls.task == "regression":
        out = Dense(1, activation="linear", name = "out")(x)
        loss = "mse"
    elif cls.task == "binary":
        out = Dense(1, activation='sigmoid', name = 'out')(x)
        loss = "binary_crossentropy"
    elif cls.task == "multiclass":
        out = Dense(len(cls.target), activation='softmax', name = 'out')(x)
        loss = "categorical_crossentropy"
    model = Model(inputs=inputs, outputs=out)

    # compile
    if params['optimizer']['type'] == 'adam':
        model.compile(loss=loss, optimizer=Adam(lr=params['optimizer']['lr'], beta_1=0.9, beta_2=0.999, decay=1e-04))
    elif params['optimizer']['type'] == 'sgd':
        model.compile(loss=loss, optimizer=SGD(lr=params['optimizer']['lr'], decay=1e-6, momentum=0.9))

    # callbacks
    er = EarlyStopping(patience=10, min_delta=1e-4, restore_best_weights=True, monitor='val_loss')
    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
    history = model.fit(train_set['X'], train_set['y'], callbacks=[er, ReduceLR],
                        epochs=params['epochs'], batch_size=params['batch_size'],
                        validation_data=[val_set['X'], val_set['y']])        
        
    # permutation importance to get a feature importance (off in default)
    # fi = PermulationImportance(model, train_set['X'], train_set['y'], cls.features)
    fi = np.zeros(len(cls.features)) # no feature importance computed

    return model, fi


