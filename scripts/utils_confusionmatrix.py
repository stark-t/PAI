from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion Matrix Function:
def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(12,9), filename=None, filename_tex=None, plot=None):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]

    # fix confusion matrix for plot
    cm = confusion_matrix(y_true, y_pred)
    if plot is not None:
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)

        cm_pd = pd.DataFrame(cm_perc, index=labels, columns=labels)
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm_pd, annot=annot, fmt='', ax=ax, cmap='Blues', square=True)
        ax.set_xlabel('\nPredicted')
        ax.set_ylabel('Actual\n')
        ax.figure.tight_layout()
        ax.figure.subplots_adjust(bottom=0.2)
        if filename is not None:
            plt.savefig(filename)
        plt.show(block=False)
        plt.pause(1)  # 3 seconds, I use 1 usually
        plt.close("all")

    if filename_tex is not None:
        with open(filename_tex, 'w') as f:
            f.write(
                '{{{:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}}}, \n'
                '{{{:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}}}, \n'
                '{{{:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}}}, \n'
                '{{{:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}}}, \n'
                '{{{:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}}}, \n'
                '{{{:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}}}, \n'
                '{{{:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}}}, \n'
                '{{{:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}}}, \n'
                '{{{:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}}},'.format(
                    cm[0, 0]/10.0, cm[0, 1]/10.0, cm[0, 2]/10.0, cm[0, 3]/10.0, cm[0, 4]/10.0, cm[0, 5]/10.0, cm[0, 6]/10.0, cm[0, 7]/10.0, cm[0, 8]/10.0,
                    cm[1, 0]/10.0, cm[1, 1]/10.0, cm[1, 2]/10.0, cm[1, 3]/10.0, cm[1, 4]/10.0, cm[1, 5]/10.0, cm[1, 6]/10.0, cm[1, 7]/10.0, cm[1, 8]/10.0,
                    cm[2, 0]/10.0, cm[2, 1]/10.0, cm[2, 2]/10.0, cm[2, 3]/10.0, cm[2, 4]/10.0, cm[2, 5]/10.0, cm[2, 6]/10.0, cm[2, 7]/10.0, cm[2, 8]/10.0,
                    cm[3, 0]/10.0, cm[3, 1]/10.0, cm[3, 2]/10.0, cm[3, 3]/10.0, cm[3, 4]/10.0, cm[3, 5]/10.0, cm[3, 6]/10.0, cm[3, 7]/10.0, cm[3, 8]/10.0,
                    cm[4, 0]/10.0, cm[4, 1]/10.0, cm[4, 2]/10.0, cm[4, 3]/10.0, cm[4, 4]/10.0, cm[4, 5]/10.0, cm[4, 6]/10.0, cm[4, 7]/10.0, cm[4, 8]/10.0,
                    cm[5, 0]/10.0, cm[5, 1]/10.0, cm[5, 2]/10.0, cm[5, 3]/10.0, cm[5, 4]/10.0, cm[5, 5]/10.0, cm[5, 6]/10.0, cm[5, 7]/10.0, cm[5, 8]/10.0,
                    cm[6, 0]/10.0, cm[6, 1]/10.0, cm[6, 2]/10.0, cm[6, 3]/10.0, cm[6, 4]/10.0, cm[6, 5]/10.0, cm[6, 6]/10.0, cm[6, 7]/10.0, cm[6, 8]/10.0,
                    cm[7, 0]/10.0, cm[7, 1]/10.0, cm[7, 2]/10.0, cm[7, 3]/10.0, cm[7, 4]/10.0, cm[7, 5]/10.0, cm[7, 6]/10.0, cm[7, 7]/10.0, cm[7, 8]/10.0,
                    cm[8, 0]/10.0, cm[8, 1]/10.0, cm[8, 2]/10.0, cm[8, 3]/10.0, cm[8, 4]/10.0, cm[8, 5]/10.0, cm[8, 6]/10.0, cm[8, 7]/10.0, cm[8, 8]/10.0
                ))

    return cm
