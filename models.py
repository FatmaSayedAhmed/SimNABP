
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import numpy as np
from transfomer import Transformer_Merged, Transformer_CorssAttention


Nb_pool_size= 5
Ag_pool_size= 54
Nb_strides=Nb_pool_size
Ag_strides=Ag_pool_size

Nb_kernal_size = 10
Ag_kernal_size = 20

num_filters_1 = 72
num_filters_2 = 100
num_filters_3 = 150
np.random.seed(123)



# Simple conv network (one 1D convolution layer)
def get_model_1ConvLayerNet(max_len_Nb, max_len_Ag, nwords, emb_dim, trainable, embedding_matrix= None):

    nanobodies = Input(shape=(max_len_Nb,))
    antigens = Input(shape=(max_len_Ag,))

    emb_Nb = Embedding(nwords, emb_dim,
                     weights=[embedding_matrix],trainable=trainable)(nanobodies) # one-hot, blosum, or Prot2vec embedding layer for antibodies
    emb_Ag = Embedding(nwords, emb_dim,
                     weights=[embedding_matrix],trainable=trainable)(antigens) # one-hot, blosum, or Prot2vec embedding layer for antigens


    Nb_conv_layer = Conv1D(filters=num_filters_1,
                                 kernel_size=Nb_kernal_size,
                                 padding="valid",
                                 activation='relu')(emb_Nb)

    Nb_max_pool_layer = MaxPooling1D(pool_size=Nb_pool_size, strides=Nb_strides)(Nb_conv_layer)

    Ag_conv_layer = Conv1D(filters=num_filters_1,
                                 kernel_size=Ag_kernal_size,
                                 padding="valid",
                                 activation='relu')(emb_Ag)

    Ag_max_pool_layer = MaxPooling1D(pool_size=Ag_pool_size, strides=Ag_strides)(Ag_conv_layer)

    # merge
    merge=Concatenate(axis=1)([Nb_max_pool_layer, Ag_max_pool_layer])

    bn=BatchNormalization()(merge)

    dt=Dropout(0.5)(bn)

    Gmaxpool = GlobalMaxPooling1D()(dt)

    merge2 = Dense(50)(Gmaxpool)

    bn2=BatchNormalization()(merge2)

    acti = Activation('relu')(bn2)

    preds = Dense(1, activation='sigmoid')(acti)

    model = Model([nanobodies, antigens], preds)

    opt = tf.keras.optimizers.Nadam()

    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model

# Simple conv network (two 1D convolution layers)
def get_model_2ConvLayersNet(max_len_Nb, max_len_Ag, nwords, emb_dim, trainable, embedding_matrix= None):

    nanobodies = Input(shape=(max_len_Nb,))
    antigens = Input(shape=(max_len_Ag,))

    emb_Nb = Embedding(nwords, emb_dim,
                     weights=[embedding_matrix],trainable=trainable)(nanobodies) # one-hot, blosum, or Prot2vec embedding layer for antibodies
    emb_Ag = Embedding(nwords, emb_dim,
                     weights=[embedding_matrix],trainable=trainable)(antigens) # one-hot, blosum, or Prot2vec embedding layer for antigens


    Nb_conv_layer_1 = Conv1D(filters=num_filters_1,
                                 kernel_size=Nb_kernal_size,
                                 padding="valid",
                                 activation='relu')(emb_Nb)

    Nb_conv_layer_2 = Conv1D(filters=num_filters_2,
                                 kernel_size=Nb_kernal_size,
                                 padding="valid",
                                 activation='relu')(Nb_conv_layer_1)

    Nb_max_pool_layer = MaxPooling1D(pool_size=Nb_pool_size, strides=Nb_strides)(Nb_conv_layer_2)

    Ag_conv_layer_1 = Conv1D(filters=num_filters_1,
                                 kernel_size=Ag_kernal_size,
                                 padding="valid",
                                 activation='relu')(emb_Ag)

    Ag_conv_layer_2 = Conv1D(filters=num_filters_2,
                                 kernel_size=Ag_kernal_size,
                                 padding="valid",
                                 activation='relu')(Ag_conv_layer_1)

    Ag_max_pool_layer = MaxPooling1D(pool_size=Ag_pool_size, strides=Ag_strides)(Ag_conv_layer_2)

    # merge
    merge=Concatenate(axis=1)([Nb_max_pool_layer, Ag_max_pool_layer])

    bn=BatchNormalization()(merge)

    dt=Dropout(0.5)(bn)

    Gmaxpool = GlobalMaxPooling1D()(dt)

    merge2 = Dense(50)(Gmaxpool)

    bn2=BatchNormalization()(merge2)

    acti = Activation('relu')(bn2)

    preds = Dense(1, activation='sigmoid')(acti)

    model = Model([nanobodies, antigens], preds)

    opt = tf.keras.optimizers.Nadam()

    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model

# Simple conv network (three 1D convolution layers)
def get_model_3ConvLayersNet(max_len_Nb, max_len_Ag, nwords, emb_dim, trainable, embedding_matrix= None):

    nanobodies = Input(shape=(max_len_Nb,))
    antigens = Input(shape=(max_len_Ag,))

    emb_Nb = Embedding(nwords, emb_dim,
                     weights=[embedding_matrix],trainable=trainable)(nanobodies) # one-hot, blosum, or Prot2vec embedding layer for antibodies
    emb_Ag = Embedding(nwords, emb_dim,
                     weights=[embedding_matrix],trainable=trainable)(antigens) # one-hot, blosum, or Prot2vec embedding layer for antigens


    Nb_conv_layer_1 = Conv1D(filters=num_filters_1,
                                 kernel_size=Nb_kernal_size,
                                 padding="valid",
                                 activation='relu')(emb_Nb)

    Nb_conv_layer_2 = Conv1D(filters=num_filters_2,
                                 kernel_size=Nb_kernal_size,
                                 padding="valid",
                                 activation='relu')(Nb_conv_layer_1)

    Nb_conv_layer_3 = Conv1D(filters=num_filters_3,
                                 kernel_size=Nb_kernal_size,
                                 padding="valid",
                                 activation='relu')(Nb_conv_layer_2)

    Nb_max_pool_layer = MaxPooling1D(pool_size=Nb_pool_size, strides=Nb_strides)(Nb_conv_layer_3)

    Ag_conv_layer_1 = Conv1D(filters=num_filters_1,
                                 kernel_size=Ag_kernal_size,
                                 padding="valid",
                                 activation='relu')(emb_Ag)

    Ag_conv_layer_2 = Conv1D(filters=num_filters_2,
                                 kernel_size=Ag_kernal_size,
                                 padding="valid",
                                 activation='relu')(Ag_conv_layer_1)

    Ag_conv_layer_3 = Conv1D(filters=num_filters_3,
                                 kernel_size=Ag_kernal_size,
                                 padding="valid",
                                 activation='relu')(Ag_conv_layer_2)

    Ag_max_pool_layer = MaxPooling1D(pool_size=Ag_pool_size, strides=Ag_strides)(Ag_conv_layer_3)

    # merge
    merge=Concatenate(axis=1)([Nb_max_pool_layer, Ag_max_pool_layer])

    bn=BatchNormalization()(merge)

    dt=Dropout(0.5)(bn)

    Gmaxpool = GlobalMaxPooling1D()(dt)

    merge2 = Dense(50)(Gmaxpool)

    bn2=BatchNormalization()(merge2)

    acti = Activation('relu')(bn2)

    preds = Dense(1, activation='sigmoid')(acti)

    model = Model([nanobodies, antigens], preds)

    opt = tf.keras.optimizers.Nadam()

    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model