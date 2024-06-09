import os
import pandas as pd
import numpy as np
import torch
import re

import scienceplots
import matplotlib as mat
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

order = 6
beta = 10.0

l_ = 2
c_ = 32
b_ = 8

cdict = {
    "blue": "#0077BB",
    "cyan": "#33BBEE",
    "teal": "#009988",
    "orange": "#EE7733",
    "red": "#CC3311",
    "magenta": "#EE3377",
    "grey": "#BBBBBB",
}
plt.switch_backend("TkAgg")
plt.style.use(["science", "std-colors"])
mat.rcParams["font.size"] = 16
mat.rcParams["mathtext.fontset"] = "cm"
mat.rcParams["font.family"] = "Times New Roman"
# size = 36
# sizein = 12

colors = [
    cdict["blue"],
    cdict["red"],
    cdict["orange"],
    cdict["magenta"],
    cdict["cyan"],
    "black",
    cdict["teal"],
    "grey",
]
pts = ["s", "^", "v", "p", "s", "o", "d"]


def plot_hist(
    order, beta, dims=[0, 1], has_weight=True, num_bins=25, xlabel="variable"
):
    # fig, ax = plt.subplots(figsize=(5, 5))
    plt.figure(figsize=(8, 7))
    if has_weight:
        hist = torch.load(
            "histogramWeight_o{0}_beta{1}_l{2}c{3}b{4}.pt".format(
                order, beta, l_, c_, b_
            )
        ).numpy()
        figname = (
            "histogramWeight_o{0}_beta{1}_l{2}c{3}b{4}".format(order, beta, l_, c_, b_)
            + xlabel
            + ".pdf"
            # "histogramWeightVegas_o{0}_beta{1}".format(order, beta) + xlabel + ".pdf"
        )
        plt.ylabel("weighted density distribution")
    else:
        hist = torch.load(
            "histogram_o{0}_beta{1}_l{2}c{3}b{4}.pt".format(order, beta, l_, c_, b_)
        ).numpy()
        figname = (
            "histogram_o{0}_beta{1}_l{2}c{3}b{4}".format(order, beta, l_, c_, b_)
            + xlabel
            + ".pdf"
        )
        # hist = torch.load("histogramVegas_o{0}_beta{1}.pt".format(order, beta)).numpy()
        # figname = "histogramVegas_o{0}_beta{1}".format(order, beta) + xlabel + ".pdf"
        plt.ylabel("density distribution")

    bins = np.linspace(0, 1, num_bins + 1)

    for d in dims:
        plt.stairs(hist[:, d], bins, label="{0} Dim".format(d))
    if xlabel == "rescaled_p":
        plt.xlabel(r"rescaled $p$")
    elif xlabel == "tau":
        plt.xlabel(r"$\tau$")
    elif xlabel == "theta":
        plt.xlabel(r"$\theta/\pi$")
    elif xlabel == "phi":
        plt.xlabel(r"$\phi/2\pi$")
    # plt.title("Histogram of learned distribution")
    plt.legend(loc="best", fontsize=14)
    plt.savefig(figname)


if __name__ == "__main__":
    ind_tau = np.arange(order - 1)
    ind_p = np.arange(order - 1, 2 * order - 1)
    ind_theta = np.arange(2 * order - 1, 3 * order - 1)
    ind_phi = np.arange(3 * order - 1, 4 * order - 1)
    plot_hist(order, beta, ind_tau, True, xlabel="tau")
    plot_hist(order, beta, ind_tau, False, xlabel="tau")
    plot_hist(order, beta, ind_p, True, xlabel="rescaled_p")
    plot_hist(order, beta, ind_p, False, xlabel="rescaled_p")
    plot_hist(order, beta, ind_theta, True, xlabel="theta")
    plot_hist(order, beta, ind_theta, False, xlabel="theta")
    plot_hist(order, beta, ind_phi, True, xlabel="phi")
    plot_hist(order, beta, ind_phi, False, xlabel="phi")
