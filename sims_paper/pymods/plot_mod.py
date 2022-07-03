import time
import pathlib
from glob import glob

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import plotly.colors as pcol
import cosmoglobe
from tqdm import tqdm 


def get_gaussian_dist(x, mu=0, sigma=1):
    """
    Method returns Gaussian distribution for given mu and sigma
    """
    return np.exp((-(x-mu)**2)/(2*sigma**2)) / np.sqrt(2*np.pi*sigma**2) 

def plot_component_maps(plots_dir, plot_fname, map_input, map_mu, 
        map_rms, map_output, component="CMB"):
    """
    Method for plotting component maps. 
    """
    #m_matrix = np.array([input_map, mean_map, rms_map])
    map_tot = [map_input, map_mu, map_rms, map_output]
    idx = 1
    stokes_params = ["I", "Q", "U"]
    titles  = [
               r"$\mathrm{Input}$", 
               r"$\mathrm{Mean}$", 
               r"$\mathrm{STD}$", 
               r"$\frac{\mathrm{Mean - Input}}{\mathrm{STD}}$"
              ]
    for row, ti in enumerate(titles):#range(len(titles)):
        for col, sig in enumerate(stokes_params):#range(len(stokes_params)):
            sub = (len(titles), len(stokes_params), idx)
            # subplots(nrows, ncols, ...)
            cosmoglobe.plot(map_tot[row][col][:], 
                    llabel = sig, 
                    rlabel = ti, 
                    sub = sub, 
                    width  = 12, 
                    unit = "$\mathrm{\mu K_{CMB}}$"
                    )
            idx += 1
    plt.tight_layout()
    plt.savefig(plots_dir.joinpath(plot_fname).resolve())
    #plt.show()
    plt.close()

def plot_component_hists(plots_dir, plot_fname, map_output):
    """
    Method for plotting component histograms.
    """
    x = np.linspace(-5, 5, 10000)
    y = get_gaussian_dist(x, 0, 1)

    stokes_params = ["I", "Q", "U"]
    figure = plt.figure(figsize=(15,5))
    cols = len(stokes_params)
    ax = [1 for i in range(cols)]
    idx = 1
    for col in range(cols):
        ax[col] = figure.add_subplot(1, cols, idx)
        ax[col].set_title(f"{stokes_params[col]}")
        ax[col].hist(map_output[col], color="C10", bins=100, range=(-6, 6),
                alpha=1, lw=0.1, 
                density=True, rasterized=True)
        ax[col].plot(x,y, "--", alpha=0.8, color="C3", 
                rasterized=True, label=f"$N(0,1)$")
        idx += 1
    plt.savefig(plots_dir.joinpath(plot_fname).resolve())
    #plt.show()
    plt.close()
