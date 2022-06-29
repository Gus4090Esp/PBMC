from load_plotf import *
from load_data import *


## alright we need to create some method to expediate
## the graphing of our data and updating figures

## we are going to take the gromacs approach.
## we will store the title of our data with their
## corresponding ploting methods
input1 = [make_plots, PC[:,0], PC[:,1], CCPSA_path, 'Principal Component 1', 'Principal Component 2', category_labels, 'Categorizing Cell Types by PSA', categories];
input2 = [make_plots, PC[:,0], PC[:,1], GPCAKM_path, 'Principal Component 1', 'Principal Component 2', pca_kmeans_labels, 'Grouping PCA by KMeans Clustering'];
input3 = [make_plots, TS[:,0], TS[:,1], CCTSNE_path, 'Component 1', 'Component 2', category_labels, 'Categorizing Cell Types By TSNE', categories];
input4 = [make_plots, TS[:,0], TS[:,1], GTSNEK_path, 'Component 1', 'Component 2', tsne_kmeans_labels, 'Grouping TSNE by KMeans Clustering'];
input5 = [make_plots, vae_z_mean[:,0], vae_z_mean[:,1], CCVAE_path, 'Component 1', 'Component 2', category_labels, CCVAE_title, categories];
input6 = [make_loss_plot, range(len(vae_loss)), vae_loss, "VAE", fig_file_path + "VAE_Loss"];
input7 = [make_plots, ae_zz[:,0], ae_zz[:,1], CCA1_path, 'Component 1', 'Component 2', category_labels, CCA1_title, categories];
input8 = [make_loss_plot, range(len(A1_loss)), A1_loss, "AutoEncoder", fig_file_path + "AE_Loss"];

subprompt = "Please input a number associated with the above graphs: ";
input_dict = {'Categorizing Cell Types by PSA': input1,
'Grouping PCA by KMeans Clustering': input2,
'Categorizing Cell Types By TSNE': input3,
'Grouping TSNE by KMeans Clustering': input4,
"Categorizing Cell Types Using A Variational Autoencoder": input5,
"Loss of " + "VAE": input6,
"Categorizing Cell Types Using An Autoencoder": input7,
"Loss of " + "AutoEncoder": input8,
};

prompt = "Please Choose What You Would Like To Graph \n";
prompt += "For each graph you want to update press the number \n";
prompt += "associated with data (to the left of the labels). \n";
prompt += "After each int press enter and you can input \n";
prompt += "another int to update another graph. When you are \n";
prompt += "inputting keys simply press enter twice \n";
prompt += "\n";
prompt += "\n";

for i, j in enumerate(input_dict.keys()):
    curr_str = str(i) + ": " + j + "\n";
    prompt += curr_str;

print(prompt);
input_keys = [];
nn_input_dict = len(input_dict.keys());

while True:
    x = input(subprompt);
    if not x == "":
        try:
            x = int(x);
            if x >= nn_input_dict:
                print("Oh no you inputted a number outside the range of listed graphs \n");
            else:
                input_keys.append(x);
        except ValueError:
            print("oh no dawg you did not enter a number \n");
            print("please enter a number \n")
    else:
        x = input("are you done inputting?(y/n)");
        if x == "y":
            break;

input_keys = np.array(input_keys, dtype = "int");
input_keys = np.unique(input_keys);
input_names = list(input_dict.keys());

for key in input_keys:
    curr_data = input_dict[input_names[key]];
    curr_plt = curr_data[0];
    curr_plt(curr_data[1::]);
