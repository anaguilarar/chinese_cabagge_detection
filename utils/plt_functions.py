import matplotlib.pyplot as plt
import random


def plot_multiple_rgb_images(npimages, n_images=5):
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.labelsize'] = False
    plt.rcParams['ytick.labelsize'] = False
    plt.rcParams['xtick.top'] = False
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.right'] = False
    plt.rcParams['figure.figsize'] = [20, 5]
    idlist = list(range(len(npimages)))
    random.shuffle(idlist)
    # plot images
    for i in range(n_images):
        plt.subplot(1, n_images, i + 1)
        plt.imshow(npimages[idlist[i]])
