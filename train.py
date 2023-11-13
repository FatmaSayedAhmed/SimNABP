# --------------------------------------------------------------------
# To compare with pioneers, we quoted some of the train procedure from EPIVAN
# --------------------------------------------------------------------


from models import get_model_1ConvLayerNet , get_model_2ConvLayersNet, get_model_3ConvLayersNet
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import tensorflow as tf
import os


os.environ["CUDA_VISIBLE_DEVICES"]="4"
modelNo = 5
# mod = "1ConvLayerNet"
# mod = "2ConvLayersNet"
mod = "3ConvLayersNet"

trainable = True
# trainable = False

emd_Type="protVec_trainable"
# emd_Type="protVec"

# emd_Type="blosum"
# emd_Type="blosum_trainable"

# emd_Type="oneHot_trainable"
# emd_Type="oneHot"

k = 3  # 1
emb_dim = 100
# emb_dim = 20

embedding_matrix = np.load('data/protVec_embedding.npy')
# embedding_matrix = np.load('data/oneHot_embedding.npy')
# embedding_matrix = np.load('data/blosum_embedding.npy')

# nwords = 21
nwords = 8002
batch_size=64
epochs=20
Data_dir = './data/'
max_len_Nb = 175
max_len_Ag = 1816

# model = get_model_1ConvLayerNet(max_len_Nb, max_len_Ag, nwords, emb_dim, trainable, embedding_matrix)
# model = get_model_2ConvLayersNet(max_len_Nb, max_len_Ag, nwords, emb_dim, trainable, embedding_matrix)
model = get_model_3ConvLayersNet(max_len_Nb, max_len_Ag, nwords, emb_dim, trainable, embedding_matrix)

Nb_Ag_pairs_tra = np.load(Data_dir+'Nb_Ag_pairs_'+str(k)+'Kmer_train.npz') # load the training data

X_Nb_tra, X_Ag_tra, y_tra = Nb_Ag_pairs_tra['X_Nb_tra'], Nb_Ag_pairs_tra['X_Ag_tra'], Nb_Ag_pairs_tra['y_tra']

X_Nb_tra, X_Nb_val, X_Ag_tra, X_Ag_val, y_tra, y_val = train_test_split(
    X_Nb_tra, X_Ag_tra, y_tra, test_size=0.05,stratify=y_tra, random_state=250) # split train dataset into train and validation

opt = tf.keras.optimizers.Nadam()
model.compile(loss='binary_crossentropy', optimizer=opt,
              metrics=['acc'])
model.summary()

checkpoint_path = "models/" + mod + "/" + emd_Type + "/" +str(modelNo) + "/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


# Create a callback that saves the model's weights every 1 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=5*batch_size)

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

history = model.fit([X_Nb_tra, X_Ag_tra], y_tra,
                    validation_data=([X_Nb_val, X_Ag_val], y_val),
                    epochs=epochs, batch_size=batch_size,
                    callbacks=[cp_callback],
                    verbose=0
                    )

print("****************Training Done ****************")
