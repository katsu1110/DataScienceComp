import numpy as np
import pandas as pd
import os, sys
import math

# keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import utils
from tensorflow.keras import layers

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
	return layers.Lambda(lambda x: x*K.tanh(K.softplus(x)))(x)

utils.get_custom_objects().update({'mish': layers.Activation(mish)})

def nn_model(cls, train_set, val_set):
    """
    NN hyperparameters and models
    """

    # adapted from https://github.com/ghmagazine/kagglebook/blob/master/ch06/ch06-03-hopt_nn.py
    params = {
        'input_dropout': 0.0,
        'hidden_layers': 2,
        'hidden_units': 64,
        'embedding_out_dim': 4,
        'hidden_activation': 'mish', 
        'hidden_dropout': 0.04,
        'gauss_noise': 0.01,
        'norm_type': 'batch', # layer
        'optimizer': {'type': 'adam', 'lr': 1e-3},
        'batch_size': 64,
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
            input_ = layers.Input(shape=(1,))
            embedding = layers.Embedding(int(np.absolute(cls.train_df[i]).max() + 1), embedding_out_dim, input_length=1)(input_)
            embedding = layers.Reshape(target_shape=(embedding_out_dim,))(embedding)
            inputs.append(input_)
            embeddings.append(embedding)
        input_numeric = layers.Input(shape=(len(cls.features) - len(cls.categoricals),))
        embedding_numeric = layers.Dense(n_neuron, activation=params['hidden_activation'])(input_numeric)
        inputs.append(input_numeric)
        embeddings.append(embedding_numeric)
        x = layers.Concatenate()(embeddings)

    else: # no categorical features
        inputs = layers.Input(shape=(len(cls.features), ))
        x = layers.Dense(n_neuron, activation=params['hidden_activation'])(inputs)
        x = layers.Dropout(params['hidden_dropout'])(x)
        x = layers.GaussianNoise(params['gauss_noise'])(x)
        if params['norm_type'] == 'batch':
            x = layers.BatchNormalization()(x)
        elif params['norm_type'] == 'layer':
            x = layers.LayerNormalization()(x)
        
    # more layers
    for i in np.arange(params['hidden_layers'] - 1):
        x = layers.Dense(n_neuron // (2 * (i+1)), activation=params['hidden_activation'])(x)
        x = layers.Dropout(params['hidden_dropout'])(x)
        x = layers.GaussianNoise(params['gauss_noise'])(x)
        if params['norm_type'] == 'batch':
            x = layers.BatchNormalization()(x)
        elif params['norm_type'] == 'layer':
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
    if params['optimizer']['type'] == 'adam':
        model.compile(loss=loss, optimizer=optimizers.Adam(lr=params['optimizer']['lr'], beta_1=0.9, beta_2=0.999, decay=params['optimizer']['lr']/100))
    elif params['optimizer']['type'] == 'sgd':
        model.compile(loss=loss, optimizer=optimizers.SGD(lr=params['optimizer']['lr'], decay=1e-6, momentum=0.9))

    # callbacks
    er = callbacks.EarlyStopping(patience=8, min_delta=params['optimizer']['lr'], restore_best_weights=True, monitor='val_loss')
    ReduceLR = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=1, epsilon=params['optimizer']['lr'], mode='min')
    history = model.fit(train_set['X'], train_set['y'], callbacks=[er, ReduceLR],
                        epochs=params['epochs'], batch_size=params['batch_size'],
                        validation_data=[val_set['X'], val_set['y']])        
        
    # permutation importance to get a feature importance (off in default)
    # fi = PermulationImportance(model, train_set['X'], train_set['y'], cls.features)
    fi = np.zeros(len(cls.features)) # no feature importance computed

    return model, fi


