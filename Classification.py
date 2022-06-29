from load_data import *
from load_plotf import *
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score
import graphviz
from load_data import *



input_shape = len(X_train[0]);
output_shape = len(y_train[0]);
N_iter = 300;

## decision trees are great for classification
## however they can easily over fit data
## to remedy the situation we are going to
## prune the tree to maximize efficiency

clf = tree.DecisionTreeClassifier(max_depth = 6);
clf.fit(X_train, y_train);
pp = clf.cost_complexity_pruning_path(X_train, y_train);
ccp_alphas, impurities = pp.ccp_alphas, pp.impurities;

clfs = [];
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(random_state = 0, ccp_alpha = ccp_alpha, max_depth = 6);
    clf.fit(X_train, y_train);
    clfs.append(clf);
clfs = clfs[:-1];
ccp_alphas = ccp_alphas[:-1]
node_counts = [clf.tree_.node_count for clf in clfs];
depth = [clf.tree_.max_depth for clf in clfs];


## lassification using a feed forward
## neural network

nodes = int(np.sqrt(input_shape)) - (int(np.sqrt(input_shape)) % 4);
input_layer = keras.Input(shape = (input_shape, ));
hidden_layer_1 = layers.Dense(nodes, activation = "relu")(input_layer);
hidden_layer_2 = layers.Dense(nodes, activation = "relu")(hidden_layer_1);
hidden_layer_3 = layers.Dense(nodes, activation = "relu")(hidden_layer_2);
output_layer = layers.Dense(output_shape, activation = "softmax")(hidden_layer_3);
neural_net = keras.Model(input_layer, output_layer, name = "neural_net")
neural_net.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"]);
neural_net.fit(X_train, y_train, epochs = N_iter);
NNA = neural_net.history.history['accuracy'];
NNL = neural_net.history.history['loss'];
y_pred = neural_net.predict(X);
