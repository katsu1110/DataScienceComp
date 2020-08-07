import numpy as np
import pandas as pd
import os, sys
import math
import random
from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict

# keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras import layers

def seed_everything(seed : int) -> NoReturn :    
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

def nn_model(cls, train_set, val_set):
    """
    NN hyperparameters and models
    """

    # set seed for tf
    seed_everything(cls.seed)

    # adapted from https://github.com/ghmagazine/kagglebook/blob/master/ch06/ch06-03-hopt_nn.py
    if not cls.params:
        params = {
            'input_dropout': 0.0,
            'hidden_layers': 3,
            'hidden_units': 256,
            'embedding_out_dim': 4,
            'hidden_activation': 'relu', 
            'hidden_dropout': 0.04,
            'gauss_noise': 0.01,
            'norm_type': 'batch', # layer
            'optimizer': {'type': 'adam', 'lr': 1e-3},
            'batch_size': 256,
            'epochs': 40
        }
        cls.params = params

    # NN model architecture
    inputs = []
    n_neuron = cls.params['hidden_units']

    # embedding for categorical features 
    if len(cls.categoricals) > 0:
        embeddings = []
        embedding_out_dim = cls.params['embedding_out_dim']
        for i in cls.categoricals:
            input_ = layers.Input(shape=(1,))
            embedding = layers.Embedding(int(np.absolute(cls.train_df[i]).max() + 1), embedding_out_dim, input_length=1)(input_)
            embedding = layers.Reshape(target_shape=(embedding_out_dim,))(embedding)
            inputs.append(input_)
            embeddings.append(embedding)
        input_numeric = layers.Input(shape=(len(cls.features) - len(cls.categoricals),))
        embedding_numeric = layers.Dense(n_neuron, activation=cls.params['hidden_activation'])(input_numeric)
        inputs.append(input_numeric)
        embeddings.append(embedding_numeric)
        x = layers.Concatenate()(embeddings)

    else: # no categorical features
        inputs = layers.Input(shape=(len(cls.features), ))
        x = layers.Dense(n_neuron, activation=cls.params['hidden_activation'])(inputs)
        x = layers.Dropout(cls.params['hidden_dropout'])(x)
        x = layers.GaussianNoise(cls.params['gauss_noise'])(x)
        if cls.params['norm_type'] == 'batch':
            x = layers.BatchNormalization()(x)
        elif cls.params['norm_type'] == 'layer':
            x = layers.LayerNormalization()(x)
        
    # more layers
    for i in np.arange(cls.params['hidden_layers'] - 1):
        x = layers.Dense(n_neuron // (2 * (i+1)), activation=cls.params['hidden_activation'])(x)
        x = layers.Dropout(cls.params['hidden_dropout'])(x)
        x = layers.GaussianNoise(cls.params['gauss_noise'])(x)
        if cls.params['norm_type'] == 'batch':
            x = layers.BatchNormalization()(x)
        elif cls.params['norm_type'] == 'layer':
            x = layers.LayerNormalization()(x)
    
    # output
    if cls.task == "regression":
        out = layers.Dense(1, activation="linear", name = "out")(x)
        loss = "mse"
    elif cls.task == "binary":
        out = layers.Dense(1, activation='sigmoid', name = 'out')(x)
        loss = "binary_crossentropy"
    elif cls.task == "multiclass":
        out = layers.Dense(len(np.unique(cls.train_df[cls.target].values)), activation='softmax', name = 'out')(x)
        loss = "categorical_crossentropy"
    model = models.Model(inputs=inputs, outputs=out)

    # compile
    if cls.params['optimizer']['type'] == 'adam':
        model.compile(loss=loss, optimizer=optimizers.Adam(lr=cls.params['optimizer']['lr'], beta_1=0.9, beta_2=0.999, decay=cls.params['optimizer']['lr']/100))
    elif cls.params['optimizer']['type'] == 'sgd':
        model.compile(loss=loss, optimizer=optimizers.SGD(lr=cls.params['optimizer']['lr'], decay=1e-6, momentum=0.9))

    # callbacks
    early_stop = callbacks.EarlyStopping(patience=8, min_delta=cls.params['optimizer']['lr'], restore_best_weights=True, monitor='val_loss')
    lr_schedule = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=1, epsilon=cls.params['optimizer']['lr'], mode='min')
    history = model.fit(train_set['X'], train_set['y'], callbacks=[early_stop, lr_schedule],
                        epochs=cls.params['epochs'], batch_size=cls.params['batch_size'],
                        validation_data=(val_set['X'], val_set['y']))        
        
    fi = np.zeros(len(cls.features)) # no feature importance computed

    return model, fi


