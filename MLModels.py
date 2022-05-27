import os, pickle, datetime
import tensorflow as tf

import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Reshape, Activation, concatenate, LSTM, ConvLSTM1D, GRU, Bidirectional, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, AveragePooling2D, Conv1D, Lambda, TimeDistributed, Add

from MetricsBinarized import *
from DataGenerator import DataGenerator



def identity_block(x, filters, kernel_sizes):   
    x_skip = x
    
    x = Conv2D(filters, kernel_sizes, padding = 'same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, kernel_sizes, padding = 'same')(x)
    x = BatchNormalization(axis=3)(x)
    
    x = Add()([x, x_skip])     
    x = Activation('relu')(x)
    return x
    
def convolutional_block(x, filters, kernel_sizes):   
    x_skip = x
    x = Conv2D(filters, kernel_sizes, padding='same', strides=(2,2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, kernel_sizes, padding = 'same')(x)
    x = BatchNormalization(axis=3)(x)
    
    x_skip = Conv2D(filters, (1,1), strides=(2,2))(x_skip)
    
    x = Add()([x, x_skip])     
    x = Activation('relu')(x)
    return x








def build_model(base_model_type='conv_net',                   # name of one of the predefined models, see below
                dataGenerator=None,                           # DataGenerator to optionally fit the input and output shapes to
                input_shape=(252, 9, 1),                      # base input shape (not including optional MFCC or analytic results
                output_shape=(48),                            # output shape
                loss_fn=MLMetrics().full_loss,                # loss function to be minimized
                mfcc_size=None,                               # optional MFCC input's feature size in case they are passed with the input
                include_analytic_results=False,               # whether analytic result labels will be passed to the model with the input
                include_rms=False,                            # whether moving RMS results will be passed to the model along with the input
                learning_rate=0.001,                          # learning rate of the optimizer
                ):
  
  if dataGenerator is not None:           # make model according to the data generator setup
    input_shape  = (*dataGenerator.x_shape, 1)
    output_shape = dataGenerator.y_shape
    
    mfcc_size = dataGenerator.mfcc_size if hasattr(dataGenerator, 'mfcc_size') else None
    include_analytic_results = True if hasattr(dataGenerator, 'analytic_predictions') else False
    include_rms = True if hasattr(dataGenerator, 'rms') else False
  
  
  print(f'Building {base_model_type} type model')
  
  
  input_spectra = Input(shape=input_shape)
  inputs=[input_spectra]

  if base_model_type == 'neural_net':
    x = Dense(128, activation='relu')(input_spectra)
    x = Dropout(0.5)(x)
    x = Flatten()(x)


  if base_model_type == 'neural_net2':
    x = Flatten()(input_spectra)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)


  if base_model_type == 'deep_net':
    x = Dense(128, activation='relu')(input_spectra)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)





  if base_model_type == 'conv_net':
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_spectra)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)





  if base_model_type == 'conv_net2':
    x = Conv2D(32, kernel_size=(9, 9), padding='same', activation='relu')(input_spectra)
    x = Conv2D(64, kernel_size=(5, 1), activation='relu')(x)
    x = Conv2D(64, kernel_size=(4, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)



  

  if base_model_type == 'conv_net3':
    x = Conv2D(32, kernel_size=(60, 1), padding='same', activation='relu')(input_spectra)
    x = Conv2D(64, kernel_size=(12, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(5, 3))(x)
    x = Conv2D(64, kernel_size=(4, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(4, 2))(x)
    x = Conv2D(64, kernel_size=(3, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 1))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
  
  if base_model_type == 'conv_net2_avgpool':
    x = Conv2D(32, kernel_size=(12, 9), padding='same', activation='relu')(input_spectra)
    x = Conv2D(64, kernel_size=(5, 1), activation='relu')(x)
    x = Conv2D(64, kernel_size=(4, 1), activation='relu')(x)
    x = AveragePooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)


  if base_model_type == 'conv_net2_nodropout':
    x = Conv2D(32, kernel_size=(12, 9), padding='same', activation='relu')(input_spectra)
    x = Conv2D(64, kernel_size=(5, 1), activation='relu')(x)
    x = Conv2D(64, kernel_size=(4, 1), activation='relu')(x)
    x = AveragePooling2D(pool_size=(3, 3))(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)






  if base_model_type == 'conv_net4':
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_spectra)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, kernel_size=(3, 1), activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 1), activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)




  if base_model_type == 'conv_net5':
    x = Conv2D(32, kernel_size=(12, 3), activation='relu')(input_spectra)
    x = Conv2D(64, kernel_size=(12, 3), activation='relu')(x)
    x = Conv2D(64, kernel_size=(12, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(4, 2))(x)
    x = Conv2D(128, kernel_size=(6, 1), activation='relu')(x)
    x = Conv2D(128, kernel_size=(6, 1), activation='relu')(x)
    x = Conv2D(128, kernel_size=(6, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Conv2D(128, kernel_size=(6, 1), activation='relu')(x)
    x = Conv2D(128, kernel_size=(6, 1), activation='relu')(x)
    x = Conv2D(128, kernel_size=(6, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)




  if base_model_type == 'conv_net6':
    x = Conv2D(32, kernel_size=(12, 3), activation='relu')(input_spectra)
    x = Conv2D(64, kernel_size=(12, 3), activation='relu')(x)
    x = Conv2D(64, kernel_size=(12, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)




  if base_model_type == 'conv_net7':
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_spectra)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)




  if base_model_type == 'conv_net8':
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(input_spectra)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)





  if base_model_type == 'conv_net9':
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(input_spectra)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)






  if base_model_type == 'conv_net10':
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_spectra)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 1))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)


  if base_model_type == 'conv_net11':
    x = Conv2D(32, kernel_size=(5, 3), activation='relu')(input_spectra)
    x = Conv2D(64, kernel_size=(5, 3), activation='relu')(x)
    x = Conv2D(64, kernel_size=(5, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)


  if base_model_type == 'conv_net12':
    x = Conv2D(32, kernel_size=(5, 3), padding='same', activation='relu')(input_spectra)
    x = Conv2D(64, kernel_size=(5, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, kernel_size=(5, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(5, 3))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)



  if base_model_type == 'conv_net13':
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_spectra)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 1))(x)
    x = Conv2D(128, kernel_size=(3, 1), activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 1), activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 1))(x)
    x = Conv2D(256, kernel_size=(3, 1), activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 1), activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 1))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)



  if base_model_type == 'conv_net14':
    x = Conv2D(32, kernel_size=(5, 3), activation='relu')(input_spectra)
    x = Conv2D(64, kernel_size=(5, 3), activation='relu')(x)
    x = Conv2D(64, kernel_size=(5, 3), activation='relu')(x)
    x = Conv2D(64, kernel_size=(5, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(5, 1))(x)
    x = Conv2D(128, kernel_size=(3, 1), activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 1), activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 1))(x)
    x = Conv2D(256, kernel_size=(3, 1), activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 1), activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 1))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)



  if base_model_type == 'conv_net15':
    x = Conv2D(32, kernel_size=(5, 1), activation='relu')(input_spectra)
    x = Conv2D(64, kernel_size=(5, 1), activation='relu')(x)
    x = Conv2D(64, kernel_size=(5, 1), activation='relu')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=1)(x)
    x = Conv2D(64, kernel_size=(3, 1), activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 1), activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 1), activation='relu')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=1)(x)
    x = Conv2D(128, kernel_size=(3, 1), activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 1), activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 1), activation='relu')(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=1)(x)
    x = Conv2D(128, kernel_size=(3, 1), activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 1), activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 1), activation='relu')(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)
    
    
    

  if base_model_type == 'conv_net16':
    x = Conv2D(32, kernel_size=(5, 3), activation='relu')(input_spectra)
    x = Conv2D(64, kernel_size=(5, 3), activation='relu')(x)
    x = Conv2D(64, kernel_size=(5, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(1,1))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    

  if base_model_type == 'conv_net17':
    x = Conv2D(32, kernel_size=(5, 3), strides=(5,1), activation='relu')(input_spectra)
    x = Conv2D(64, kernel_size=(5, 3), strides=(5,1), activation='relu')(x)
    x = Conv2D(64, kernel_size=(5, 3), strides=(5,1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
  if base_model_type == 'conv_net18':
    x = Conv2D(32, kernel_size=(5, 3), padding="same", strides=(5,1), activation='relu')(input_spectra)
    x = Conv2D(64, kernel_size=(5, 3), padding="same", strides=(5,1), activation='relu')(x)
    x = Conv2D(64, kernel_size=(5, 3), padding="same", strides=(5,1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(4, 9))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    

  if base_model_type == 'conv_net19':
    x = Conv2D(32, kernel_size=(5, 3), activation='relu')(input_spectra)
    x = Conv2D(64, kernel_size=(5, 3), activation='relu')(x)
    x = Conv2D(64, kernel_size=(5, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Conv2D(64, kernel_size=(3, 1), activation='relu')(input_spectra)
    x = Conv2D(128, kernel_size=(3, 1), activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 1))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    

  if base_model_type == 'conv_net20':
    x = Conv2D(32, kernel_size=(12, 3), padding='same', activation='relu')(input_spectra)
    x = AveragePooling2D(pool_size=(12, 1), strides=1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(64, kernel_size=(6, 2), activation='relu')(x)
    x = AveragePooling2D(pool_size=(6, 1), strides=1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(64, kernel_size=(3, 1), activation='relu')(x)
    x = AveragePooling2D(pool_size=(6, 1), strides=1)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    

  if base_model_type == 'lstm':
    x = Conv2D(32, kernel_size=(12, 9), padding='same', activation='relu')(input_spectra)
    x = Conv2D(64, kernel_size=(5, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Conv2D(64, kernel_size=(4, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    
    
    x = Reshape((x.shape[1]*x.shape[2], x.shape[3]))(x)

    x = Bidirectional(LSTM(256))(x)
    x = BatchNormalization()(x)
    
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    


  if base_model_type == 'lstm2':
    x = Conv2D(32, kernel_size=(5, 3), activation='relu')(input_spectra)
    x = Conv2D(64, kernel_size=(5, 3), activation='relu')(x)
    x = Conv2D(64, kernel_size=(5, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)
    
    
    x = Reshape((x.shape[1]*x.shape[2], x.shape[3]))(x)

    x = Bidirectional(LSTM(256))(x)
    x = BatchNormalization()(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)




  if base_model_type == 'resnet':
    x = Conv2D(64, kernel_size=(9, 9), padding='same')(input_spectra)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2,2))(x)
    
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:  # res/conv block not needed for sub-block 1
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size, (3, 3))
        else:
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size, (3, 3))
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size, (3, 3))
                
    x = tf.keras.layers.AveragePooling2D((2,2), padding = 'same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)














  x = Dense(output_shape)(x)
  
  # optionally include MFCC spectra at the tail end of the model
  if mfcc_size:
    input_mfcc = Input(shape=mfcc_size)
    inputs.append(input_mfcc)
    
    x_mfcc = Reshape((input_mfcc.shape[1], 1))(input_mfcc)
    
    x_mfcc = Conv1D(32, kernel_size=(3), strides=1, activation='relu')(x_mfcc)
    x_mfcc = MaxPooling1D(pool_size=(3), strides=1)(x_mfcc)
    x_mfcc = Conv1D(64, kernel_size=(3), strides=1, activation='relu')(x_mfcc)
    x_mfcc = MaxPooling1D(pool_size=(3), strides=1)(x_mfcc)
    x_mfcc = Conv1D(64, kernel_size=(3), strides=1, activation='relu')(x_mfcc)
    x_mfcc = MaxPooling1D(pool_size=(3))(x_mfcc)

    x_mfcc = Flatten()(x_mfcc)
    x_mfcc = Dense(256, activation='relu')(x_mfcc)
    x_mfcc = Dropout(0.1)(x_mfcc)
    x_mfcc = Dense(output_shape, activation='relu')(x_mfcc)
    x_mfcc = Dropout(0.1)(x_mfcc)
    
    
    
    merged = concatenate([x, x_mfcc])

    x = Dense(128, activation='relu')(merged)
    x = Dropout(0.1)(x)
    x = Dense(output_shape)(x)


  # optionally include results from an analytic model at the tail end of the model
  if include_analytic_results:
    input_analytic = Input(shape=output_shape)
    inputs.append(input_analytic)
    
    x_analytic = Dense(128, activation='relu')(input_analytic)
    x_analytic = Dropout(0.5)(x_analytic)
    x_analytic = Dense(output_shape, activation='relu')(x_analytic)

    merged = concatenate([x, x_analytic])

    x = Dense(128, activation='relu')(merged)
    x = Dense(output_shape)(x)




  # optionally include results from an analytic model at the tail end of the model
  if include_rms:
    input_rms = Input(shape=(input_shape[1]))
    inputs.append(input_rms)
    
    x_rms = Dense(64, activation='relu')(input_rms)
    x_rms = Dropout(0.1)(x_rms)
    
    merged = concatenate([x, x_rms])

    x = Dense(128, activation='relu')(merged)
    x = Dense(output_shape)(x)




  x = Activation('sigmoid')(x)
  model = Model(inputs=inputs, outputs=x)

  model.compile(loss=loss_fn,
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=[pr_precision, pr_recall, pr_f1, precision, recall, f1])

  return model





class MLModel:
  def __init__(self,dataGenerator,            # main DataGenerator object that the model will use
              model_type='conv_net',          # one of the prebuild model types to use
              model_path=None,                # path to a prefitted model to optionally load
              callback_monitor="val_f1",      # metric to monitor for callbacs, including checkpointing and early stopping
              early_stop_patience=3,          # number of epochs to allow the monitored metric to stagnate before stopping the model fitting
              cross_validate=False,           # number of epochs to allow the monitored metric to stagnate before stopping the model fitting
              loss_fn=MLMetrics().full_loss,  # loss function to be minimized
               ):
    
    
    
    self.dataGenerator  = dataGenerator
    self.model_type    = model_type
    self.early_stop_patience = early_stop_patience
    
    self.path_models = (((os.path.dirname(os.path.abspath(__file__)) + os.path.sep) if '__file__' in globals() 
                      else 'gdrive/MyDrive/MSc Diploma/') + 'data' + os.path.sep + 'models' + os.path.sep + 'ML' + os.path.sep)
    
    
              
              
    self.model = build_model(base_model_type=model_type, dataGenerator=dataGenerator, loss_fn=loss_fn)
    
    if model_path is not None:
      self.load_model(self.path_models + model_path + os.path.sep)
      self.model_name = model_path + os.path.sep
    
    print(self.model.summary())
    
    if not hasattr(self, 'model_name'):
      self.model_name = (f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")}_'
              + f'{model_type}_{dataGenerator.spectrum_type}'
              + f'{("_MFCC" + str(dataGenerator.mfcc_size)) if hasattr(dataGenerator, "mfcc_size") else ""}'
              + f'{"_ANALYTIC" if hasattr(dataGenerator, "analytic_predictions") else ""}'
              + f'{"_RMS" if hasattr(dataGenerator, "rms") else ""}/')
              
              
    self.checkpoint_path = self.path_models + self.model_name + os.path.sep + 'checkpoints' + os.path.sep + "cp-{epoch:02d}-{val_f1:.2f}.hdf5"
    self.checkpoint_dir  = os.path.dirname(self.checkpoint_path)
    
    if not os.path.exists(self.path_models + self.model_name):
      os.makedirs(self.path_models + self.model_name)
      os.makedirs(self.checkpoint_dir)
    
    self.callbacks = [
      keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path, 
            monitor=callback_monitor, save_best_only=False, save_weights_only=True, save_freq='epoch', verbose=1),
      keras.callbacks.EarlyStopping(monitor=callback_monitor, patience=early_stop_patience, mode='max')
     ]
    
    
    
    
  # save model configuration and weights for later reference
  def save_model(self):
    self.model.save_weights(self.path_models + self.model_name)
    
    
  # load model configuration and weights
  def load_model(self, model_path):
    model_file = sorted(os.listdir(model_path+os.path.sep+'checkpoints'+os.path.sep))[-(self.early_stop_patience+1)]
    load_status = self.model.load_weights(model_path+os.path.sep+'checkpoints'+os.path.sep+model_file)
    print(f'Loaded model from {model_path}')
    
  
  
  # fit model to the given data generator and save fitting metrics history
  # (optionally perform k-folds cross-validation if data generator is set up that way
  def fit_model(self, max_epochs=30):
  
    for fold in range(self.dataGenerator.fold_number()):
      if self.dataGenerator.fold_number() > 1:
        print(f'Fitting on fold {fold}/{self.dataGenerator.fold_number()}')
    
    
      history = self.model.fit(x=self.dataGenerator,
                  validation_data=self.dataGenerator.get_validation_generator(),
                  epochs=max_epochs, 
                  callbacks=self.callbacks,
                  verbose=1)

      with open(self.path_models + self.model_name + 'fitting_history' + str(fold) + '.pkl', 'wb') as file:
        pickle.dump(history.history, file)
      
      self.dataGenerator.next_fold()
      
    self.save_model()
  
  # evaluate current model on 
  def evaluate_model(self, x_generator=None):
    if x_generator is None:
      x_generator = self.dataGenerator.get_test_generator()
    
    metrics = self.model.evaluate(x=x_generator)
    
    with open(self.path_models + self.model_name + 'eval_results' + '.pkl', 'wb') as file:
      pickle.dump(metrics, file)
    
  
  def predict_model(self, x_generator=None, save_preds=True, separate_chunks=True):
    if x_generator is None:
      x_generator = self.dataGenerator.get_test_generator()
    
    if not separate_chunks:
      preds = self.model.predict(x=x_generator)
    
      if save_preds:
        with open(self.path_models + self.model_name + 'predictions.pkl', 'wb') as file:
          pickle.dump(labels_from_binary(prob_to_pred(preds), x_generator.class_num, x_generator.include_empty_class), file)
    
    else:
      preds, gtruths = [], []
      for i in range(x_generator.__len__()):
        print(f'\rPredicting samples: {i+1}/{x_generator.__len__()}', end='')
        sample = x_generator.__getitem__(i)
        preds.append(labels_from_binary(prob_to_pred(self.model.predict(x=sample[0])), x_generator.class_num, x_generator.include_empty_class))
        gtruths.append(labels_from_binary(sample[1], x_generator.class_num, x_generator.include_empty_class))
    
      if save_preds:
        with open(self.path_models + self.model_name + 'prediction_chunks.pkl', 'wb') as file:
          pickle.dump({'gtruths': gtruths, 'preds': preds}, file)
    
    return preds
      



# get labels original shifted midi labels from binarized form
def labels_from_binary(bin_labels, class_num=44, includes_empty_class=False):
  labels = np.zeros((bin_labels.shape[0], 6), dtype=int)
  for frame, bin_label in enumerate(bin_labels):
    bin_label = np.nonzero(bin_label)[0] + (0 if includes_empty_class else 1)
    labels[frame,:len(bin_label)] = bin_label
  return labels
  
  
 
def voting_ensemble(models, x_generator):
  preds = np.empty((len(models), sum(x_generator.chunk_sizes), x_generator.class_num))
  
  curr_ind = 0
  for ind in range(len(x_generator.__len__())):
    batch = x_generator.__getitem__(ind)[0]
    for i, m in enumerate(models):
      curr_pred = m.predict(batch)
      preds[i,curr_ind:curr_ind+len(curr_pred)] = curr_pred
      curr_ind += len(curr_pred)
      
  return prob_to_pred(np.mean(preds, axis=0))
    
    
