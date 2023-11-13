# --------------------------------------------------------------------
# To compare with pioneers, we quoted some of the test procedure from EPIVAN
# --------------------------------------------------------------------
import os
import tensorflow as tf
from models import get_model_1ConvLayerNet , get_model_2ConvLayersNet, get_model_3ConvLayersNet
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score


os.environ["CUDA_VISIBLE_DEVICES"]="2"

model=None

modelNo = 5
mod = "3ConvLayersNet"
emd_Type="protVec_trainable"
trainable = True
embedding_matrix = np.load('data/protVec_embedding.npy')

k = 3  # 1
emb_dim = 100
# emb_dim = 20
# nwords = 21
nwords = 8002

batch_size=64
epochs=20
Data_dir = './data/'
max_len_Nb = 175
max_len_Ag = 1816

model = get_model_3ConvLayersNet(max_len_Nb, max_len_Ag, nwords, emb_dim, trainable, embedding_matrix)

checkpoint_path = "models/" + mod + "/" + emd_Type + "/" +str(modelNo) + "/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

print("****************Testing****************")
Nb_Ag_pairs_test = np.load(Data_dir+'Nb_Ag_pairs_'+str(k)+'Kmer_test.npz') # load the test data

X_Nb_test, X_Ag_test, y_test = Nb_Ag_pairs_test['X_Nb_test'], Nb_Ag_pairs_test['X_Ag_test'], Nb_Ag_pairs_test['y_test']
print("y_test: ", y_test)
y_pred = model.predict([X_Nb_test, X_Ag_test])

auc = roc_auc_score(y_test, y_pred)
print("AUC : ", auc)

aupr = average_precision_score(y_test, y_pred)
print("AUPR : ", aupr)
