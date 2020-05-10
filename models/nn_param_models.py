import numpy as np
import pandas as pd
import os, sys

# keras
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, LambdaCallback
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import losses, models, optimizers
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Input, Layer, Dense, Activation, Embedding, Concatenate, Reshape, Dropout, Add, BatchNormalization, LayerNormalization, GaussianNoise
from tensorflow.keras import backend as K
import math

# enable mish
class Mish(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        base_config = super(Mish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

def mish(x):
	return tf.keras.layers.Lambda(lambda x: x*K.tanh(K.softplus(x)))(x)

get_custom_objects().update({'mish': Activation(mish)})

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
        'hidden_activation': 'mish', 
        'hidden_dropout': 0.04,
        'norm_type': 'batch', # layer
        'optimizer': {'type': 'adam', 'lr': 1e-4},
        'batch_size': 128,
        'epochs': 40
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
        embedding_numeric = Dense(n_neuron, activation=params['hidden_activation'])(input_numeric)
        inputs.append(input_numeric)
        embeddings.append(embedding_numeric)
        x = Concatenate()(embeddings)

    else: # no categorical features
        inputs = Input(shape=(len(cls.features), ))
        x = Dense(n_neuron, activation=params['hidden_activation'])(inputs)
        x = Dropout(params['hidden_dropout'])(x)
        if params['norm_type'] == 'batch':
            x = BatchNormalization()(x)
        elif params['norm_type'] == 'layer':
            x = LayerNormalization()(x)
        
    # more layers
    for i in np.arange(params['hidden_layers'] - 1):
        x = Dense(n_neuron // (2 * (i+1)), activation=params['hidden_activation'])(x)
        x = Dropout(params['hidden_dropout'])(x)
        if params['norm_type'] == 'batch':
            x = BatchNormalization()(x)
        elif params['norm_type'] == 'layer':
            x = LayerNormalization()(x)
    
    # output
    if cls.task == "regression":
        out = Dense(1, activation="linear", name = "out")(x)
        loss = "mse"
    elif cls.task == "binary":
        out = Dense(1, activation='sigmoid', name = 'out')(x)
        loss = "binary_crossentropy"
    elif cls.task == "multiclass":
        out = Dense(len(np.unique(cls.train_df[cls.target].values)), activation='softmax', name = 'out')(x)
        loss = "categorical_crossentropy"
    model = Model(inputs=inputs, outputs=out)

    # compile
    if params['optimizer']['type'] == 'adam':
        model.compile(loss=loss, optimizer=Adam(lr=params['optimizer']['lr'], beta_1=0.9, beta_2=0.999, decay=1e-04))
    elif params['optimizer']['type'] == 'sgd':
        model.compile(loss=loss, optimizer=SGD(lr=params['optimizer']['lr'], decay=1e-6, momentum=0.9))

    # callbacks
    er = EarlyStopping(patience=8, min_delta=1e-4, restore_best_weights=True, monitor='val_loss')
    ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=1, epsilon=1e-4, mode='min')
    history = model.fit(train_set['X'], train_set['y'], callbacks=[er, ReduceLR],
                        epochs=params['epochs'], batch_size=params['batch_size'],
                        validation_data=[val_set['X'], val_set['y']])        
        
    # permutation importance to get a feature importance (off in default)
    # fi = PermulationImportance(model, train_set['X'], train_set['y'], cls.features)
    fi = np.zeros(len(cls.features)) # no feature importance computed

    return model, fi


