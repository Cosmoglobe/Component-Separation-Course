import cosmoglobe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal as signal
import plotly.colors as pcol               
from cycler import cycler 
from tqdm import tqdm
from numba import jit
import corner
import sys

palette = "Plotly"                                                                   
colors = getattr(pcol.qualitative, palette)                                          
my_cycler = cycler(color=colors)


fontsize = 14
fonts = {
        "axes.labelsize": fontsize,
        "font.size": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        #"text.usetex": True,
        'axes.prop_cycle': my_cycler
        }
plt.rcParams.update(fonts)

chain = cosmoglobe.Chain('chains_sim/chain_c0001.h5')
xi_n = chain.get('tod/030/xi_n', samples=range(2000))
print(xi_n[:, :4, 0, 3003].shape)
############# Corner plot #################


detectors = ["27M", "27S", "28M", "28S"]
for i in range(4):
    figure = corner.corner(xi_n[:, :4, i, 3003], labels=[r"$\sigma_0$", r"$f_\mathrm{knee}$", r"$\alpha$", r"$ \mathrm{A}$"],
                        show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig(f"noise_parameter_coner_plot_{detectors[i]}.pdf")

############# Burn-in #################

fig = plt.figure(figsize = (12, 12))
gs = gridspec.GridSpec(2, 2, figure = fig, wspace = 0.0, hspace = 0.0)

axes = []

for i in range(4):
    lim = [1.2 * np.min(xi_n[:, 0, i, 3003]), 1.2 * np.max(xi_n[:, 0, i, 3003])]
    axes.append(fig.add_subplot(gs[i]))
    if i == 0 or i == 2:
        #axes[i].plot((xi_n[:, 0, i, 3003] - xi_n[:, 0, i, 3003].mean()) * 1e6)
        #axes[i].set_ylim(lim[0], lim[1])
        axes[i].plot((xi_n[:, 0, i, 3003]) * 1e6)
        axes[i].set_ylabel(r"$\sigma_0$ [$\mathrm{\mu}$V]")

        if i == 0:
            yticks = axes[i].get_yticks()[2::2]
        else:
            yticks = axes[i].get_yticks()[1:-1:2]
        axes[i].set_yticks(yticks)
        axes[i].set_yticklabels(yticks, rotation = 90)
        axes[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    else:
        ax = axes[i].twinx()
        #axes[i] = ax
        #ax.plot((xi_n[:, 0, i, 3003] - xi_n[:, 0, i, 3003].mean()) * 1e6)
        ax.plot((xi_n[:, 0, i, 3003]) * 1e6)
        #ax.set_ylim(lim[0], lim[1])
        ax.set_ylabel(r"$\sigma_0$ [$\mathrm{\mu}$V]")
        if i == 3:
            yticks = ax.get_yticks()[1:-2:2]
        else:
            yticks = ax.get_yticks()[1::2]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, rotation = 90)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    axes[i].text(0.05, 0.85, detectors[i], transform=axes[i].transAxes)


plt.setp(axes[1].get_yticklabels(), visible=False)
plt.setp(axes[3].get_yticklabels(), visible=False)
axes[1].set_yticks([])
axes[3].set_yticks([])

plt.setp(axes[0].get_xticklabels(), visible=False)
plt.setp(axes[1].get_xticklabels(), visible=False)

axes[2].set_xlabel("Iterations")
axes[3].set_xlabel("Iterations")
fig.savefig("sigma0_burn_in.pdf")



############# Correlation length #################
@jit
def autocorrelation(x, length):
    acf = np.zeros(length)
    acf[0] = np.corrcoef(x, x)[0, 1]
    for i in range(1, length):
        acf[i] = np.corrcoef(x[:-i:i], x[i::i])[0, 1]
    return acf

shape = xi_n.shape

corrs = np.zeros((400, shape[2], shape[3]))


for i in tqdm(range(shape[3]), position = 0):
    for j in tqdm(range(shape[2]), position = 1, leave = False):
        corrs[:, j, i] = autocorrelation(xi_n[:, 0, j, i], 400)



mask = np.all(xi_n[:, 0, :, :] == 0, axis = (0, 1))
print(mask)
mean = np.nanmean(corrs, axis = -1)
std = np.nanstd(corrs, axis = -1)

conf_top = mean + std
conf_bottom = mean - std
print(corrs[:, 0, 0])
print(conf_top[:, 0])
print(conf_bottom[:, 0])
print(np.nanmean(corrs[0, :, :]), corrs[0, :, -1])


fig = plt.figure(figsize = (12, 12))
gs = gridspec.GridSpec(2, 2, figure = fig, wspace = 0.0, hspace = 0.0)

axes = []
for i in range(4):
    axes.append(fig.add_subplot(gs[i]))
    if i == 0 or i == 2:
        #axes[i].plot((xi_n[:, 0, i, 3003] - xi_n[:, 0, i, 3003].mean()) * 1e6)
        #axes[i].set_ylim(lim[0], lim[1])
        axes[i].fill_between(np.arange(400), conf_top[:, i], conf_bottom[:, i], alpha = 0.2)
        axes[i].plot(mean[:, i], "k")
        axes[i].set_ylabel(r"Correlation coefficient")

        if i == 0:
            yticks = axes[i].get_yticks()[2::2]
        else:
            yticks = axes[i].get_yticks()[1:-1:2]
        axes[i].set_yticks(yticks)
        axes[i].set_yticklabels(yticks, rotation = 90)
        axes[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    else:
        ax = axes[i].twinx()
        #axes[i] = ax
        #ax.plot((xi_n[:, 0, i, 3003] - xi_n[:, 0, i, 3003].mean()) * 1e6)
        ax.fill_between(np.arange(400), conf_top[:, i], conf_bottom[:, i], alpha = 0.2)
        ax.plot(mean[:, i], "k")
        #ax.set_ylim(lim[0], lim[1])
        ax.set_ylabel(r"Correlation coefficient")
        if i == 3:
            yticks = ax.get_yticks()[1:-2:2]
        else:
            yticks = ax.get_yticks()[1::2]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, rotation = 90)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    axes[i].text(0.05, 0.85, detectors[i], transform=axes[i].transAxes)


plt.setp(axes[1].get_yticklabels(), visible=False)
plt.setp(axes[3].get_yticklabels(), visible=False)
axes[1].set_yticks([])
axes[3].set_yticks([])

plt.setp(axes[0].get_xticklabels(), visible=False)
plt.setp(axes[1].get_xticklabels(), visible=False)

axes[2].set_xlabel("Iterations")
axes[3].set_xlabel("Iterations")


fig.savefig("sigma0_correlation_len.pdf")
#plt.show()

