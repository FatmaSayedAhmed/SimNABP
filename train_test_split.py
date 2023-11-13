import numpy as np
from sklearn.model_selection import train_test_split


Data_dir = './data/'
k = 3 # 1 or 2

Nb_Ag_pairs = np.load(Data_dir+'Nb_Ag_pairs_' + str(k) + 'Kmer.npz') # load the training data

X_Nb, X_Ag, y = Nb_Ag_pairs['X_Nb'], Nb_Ag_pairs['X_Ag'], Nb_Ag_pairs['y']

X_Nb_tra, X_Nb_test, X_Ag_tra, X_Ag_test, y_tra, y_test = train_test_split(
    X_Nb, X_Ag, y, test_size=0.3,stratify=y, random_state=250)


np.savez(Data_dir+'Nb_Ag_pairs_' + str(k) + 'Kmer_train.npz',X_Nb_tra=X_Nb_tra,X_Ag_tra=X_Ag_tra,y_tra=y_tra)
np.savez(Data_dir+'Nb_Ag_pairs_' + str(k) + 'Kmer_test.npz',X_Nb_test=X_Nb_test,X_Ag_test=X_Ag_test,y_test=y_test)

