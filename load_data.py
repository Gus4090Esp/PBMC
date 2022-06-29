import os
import numpy as np

fig_file_name = 'Figures';
fig_file_path = fig_file_name + "/";
Xname, yname = "X", "y";
X_train_name, y_train_name = "X_train", "y_train";
X_test_name, y_test_name = "X_test", "y_test";
cat_name = "category_labels";
pca_kmeans_name, tsne_kmeans_name = "pca_kmeans_labels", "tsne_kmeans_labels";
min_max_name = "min_max";
TS_name, PC_name = "TS", "PC";
categories_name = "categories";
CCPSA_path = fig_file_path + "Categorizing_Cell_Types_PSA";
GPCAKM_path = fig_file_path + "Grouping_PSA_KMeans";
CCTSNE_path = fig_file_path + "Categorizing_Cell_Types_TSNE";
GTSNEK_path = fig_file_path + "Grouping_TSNE_KMeans";
data_name = "Data"
data_path = data_name + "/";
X, y = np.load(data_path + Xname + ".npy"), np.load(data_path + yname + ".npy");
X_train, y_train = np.load(data_path + X_train_name + ".npy"), np.load(data_path + y_train_name + ".npy");
X_test, y_test = np.load(data_path + X_test_name + ".npy"), np.load(data_path + y_test_name + ".npy");
category_labels = np.load(data_path + cat_name + ".npy");
pca_kmeans_labels, tsne_kmeans_labels = np.load(data_path + pca_kmeans_name + ".npy"), np.load(data_path + tsne_kmeans_name + ".npy");
min_max = np.load(data_path + min_max_name + ".npy");
min_X, max_X = min_max[0], min_max[1];
TS, PC = np.load(data_path + TS_name + ".npy"), np.load(data_path + PC_name + ".npy");
categories = np.load(data_path + categories_name + ".npy", allow_pickle=True);
CCVAE_title = "Categorizing Cell Types Using A Variational Autoencoder";
CCVAE_path = fig_file_path + "Categorizing_Cell_Types_VAE";
CCA1_title = "Categorizing Cell Types Using AutoEncoder";
CCA1_path = fig_file_path + "Categorizing_Cell_Types_AE";
ae_zz_name = "ae_zz"
A1_loss_name = "A1_loss";
vae_z_mean_name = "vae_z_mean";
vae_loss_name = "vae_loss";
ae_zz = np.load(data_path + ae_zz_name + ".npy");
A1_loss = np.load(data_path + A1_loss_name + ".npy");
vae_z_mean = np.load(data_path + vae_z_mean_name + ".npy");
vae_loss = np.load(data_path + vae_loss_name + ".npy");
