import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import time
import os
from load_plotf import *

## first lets supress sklearn and tensorflow warnings ##
## I need this because of some compatibility issues ##

def warn(*args, **kwargs):
    pass;

warnings.warn = warn;


## we are going to generate a lot of figures
## we therefore need somewhere to store them
fig_file_name = 'Figures';
fig_file_path = fig_file_name + "/";
if not os.path.exists(fig_file_path):
    os.system("mkdir " + fig_file_name);


## The purpose of this script is to
## show some level of compentency of ML.
## Specifically using a high level programming
## language(python) and to show that
## have knowledge in their Machine Learning
## packages. We will apply neural networks
## for dimensionality reduction and later
## for classification.

## data was acquired from genomics 10x
## and has been processed into tables

## location of gene expression matrix
## and associated labels for a givenm
## PMBC
counts_path = "Data/processed_counts.csv";
labels_path = "Data/labels.csv";

## tables for data
labels_pd = pd.read_csv(labels_path);
counts_pd = pd.read_csv(counts_path);

## Need to process the data a bit
## lets store it in variable X
## we are going to use one hot encoding
## for classification of our labels
categories = labels_pd['bulk_labels'].unique();
X = counts_pd.to_numpy()[:,1:].astype('float32');
y = np.zeros((len(labels_pd),len(categories)))
nlabels = len(labels_pd);

## one hot encoding
for i in range(nlabels):
    cell_type = labels_pd.iloc[i]['bulk_labels'];
    pos = np.where(categories == cell_type)[0];
    y[i,pos] = 1;

## right now we have our gene expresion matrix as X
## and our one hot encoding y

## now lets perform of 80:20 split of our data
## into a training set, and a testing set

my_seed = int(time.time());
np.random.seed(my_seed);
nX = len(X);
ny = len(y);
num_perm = 10;
## lets shuffle the data a few times
## to make sure the data is truly getting random
## data
for i in range(num_perm):
    perm = np.random.permutation(nX);
    X, y = X[perm], y[perm];


## note that the preprocessing created data that
## was between -2 and 30.44. If we want to use
## a variational autoencoder or a typical autoencoder
## for dimensionality reduction using activation
## functions that are strictly positive will result
## in loss of information. Yes this inevitable in
## dimensionality reduction however minimal
## changes to the input can alleviate this.

min_X = np.min(X);
if min_X < 0.0:
    X += np.abs(min_X);
    max_X = np.max(X);
    X /= max_X;
else:
    max_X = np.max(X);
    X /= max_X;

## note that as long as we store max_X and min_X
## in our code reverse the operation to get the
## original data set. In this sense the operation
## forms a bijection
split = .8
nXsplit = int(nX*split);
nysplit = int(ny*split);
X_train, y_train = X[:nXsplit], y[:nysplit];
X_test, y_test = X[nXsplit:], y[nysplit:];
category_labels = np.where(y == 1)[1];


## Because PCA is a stable dimensionality
## reduction technique we are going to use
## it to benchmark using VAE/AE
## We're also going to use TSNE
## I dont think we have will have to
## use PCA results to input into TSNE
## but on larger data sets it will be useful
pca = PCA(n_components = 2);
PC = pca.fit_transform(X);
CCPSA_path = fig_file_path + "Categorizing_Cell_Types_PSA";


pca_kmeans = KMeans(n_clusters = 4, n_init = 5, max_iter = 5000, tol = 10**-6).fit(X);
pca_kmeans_labels = pca_kmeans.labels_;
GPCAKM_path = fig_file_path + "Grouping_PSA_KMeans";


tsne = TSNE(perplexity = 30, n_components = 2, metric = 'euclidean', learning_rate = 'auto', n_iter = 5000);
TS = tsne.fit_transform(X);
CCTSNE_path = fig_file_path + "Categorizing_Cell_Types_TSNE";



tsne_kmeans = KMeans(n_clusters = 5, n_init = 5, max_iter = 5000, tol = 10**-6).fit(X);
tsne_kmeans_labels = tsne_kmeans.labels_
GTSNEK_path = fig_file_path + "Grouping_TSNE_KMeans";

## we also need somewhere to store our data
## thats been computed throughout this script
## to keep it consistent with this code, and
## we want to know which items to save and how
Xname, yname = "X", "y";
X_train_name, y_train_name = "X_train", "y_train";
X_test_name, y_test_name = "X_test", "y_test";
cat_name = "category_labels";
pca_kmeans_name, tsne_kmeans_name = "pca_kmeans_labels", "tsne_kmeans_labels";
min_max = np.array([min_X, max_X]);
min_max_name = "min_max";
TS_name, PC_name = "TS", "PC";
categories_name = "categories";

data_name = "Data"
data_path = data_name + "/";

if not os.path.exists(data_path):
    os.system("mkdir " + data_name);

my_names = [Xname, yname, X_train_name, y_train_name, X_test_name, y_test_name,cat_name, pca_kmeans_name, tsne_kmeans_name, min_max_name, TS_name, PC_name, categories_name];
my_data = [X, y, X_train, y_train, X_test, y_test, category_labels, pca_kmeans_labels, tsne_kmeans_labels, min_max, TS, PC, categories];
nn = len(my_names);
for i in range(nn):
    dpath = data_path + my_names[i];
    if os.path.exists(dpath):
        os.system("rm " + dpath);
    np.save(dpath + ".npy", my_data[i]);
