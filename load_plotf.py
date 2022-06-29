import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix

def make_loss_plot(*args):
    xx, yy, name, curr_path = args[0];
    plt.plot(xx,yy);
    plt.xlabel("epochs");
    plt.ylabel("loss");
    plt.title("Loss of " + name);
    if os.path.exists(curr_path):
        os.system("rm " + curr_path);
    plt.savefig(curr_path);
    plt.close("all");

def make_plots(*args):
    my_inputs = args[0];
    if len(my_inputs) == 6:
        tcklbls = None
        xx, yy, curr_path, xlb, ylb, lbls, tit = my_inputs;
    else:
        xx, yy, curr_path, xlb, ylb, lbls, tit, tcklbls = my_inputs;
    xlims = [np.min(xx), np.max(xx)];
    ylims = [np.min(yy), np.max(yy)];
    plt.scatter(xx, yy, c = lbls);
    plt.xlim(xlims);
    plt.ylim(ylims);
    plt.xlabel(xlb);
    plt.ylabel(ylb);
    plt.title(tit);
    cbar = plt.colorbar();
    if not tcklbls is None:
        cbar.set_ticklabels(tcklbls);
    if os.path.exists(curr_path):
        os.system("rm " + curr_path);
    plt.savefig(curr_path);
    plt.close("all");

def plot_confusionmatrix(*args):
    yyp, yy, dom, cat = args[0];
    cf = confusion_matrix(yyp, yy)
    sns.heatmap(cf, annot = True, yticklabels = cat, xticklabels = cat, cmap = "Blues", fmt = "g");
    plt.tight_layout();
    plt.show();
    plt.close("all");
