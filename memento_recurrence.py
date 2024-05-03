import numpy as np
import scipy.stats as stats
from scipy.signal import peak_widths, find_peaks
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.collections import LineCollection
from memento import *
from scipy.optimize import curve_fit
from tqdm.contrib.itertools import product


def main():
    layer_sizes = [100, 100, 100, 100, 100, 100]
    kappas = [50, 40, 30, 10, 4]       
    # kappas = [10, 10, 10]      
    k_input = 20
    recurrence = 'cosine2'

    n_timesteps=500
    stim_on=500

    R_1 = np.linspace(-2, 5, 15)
    R_2 = np.linspace(-2, 5, 15)
    R_3 = np.linspace(-2, 5, 15)
    R_4 = np.linspace(-2, 5, 15)


    # Initialize model
    model = DynaToy(layer_sizes)
    model.W_ff_0 = 0.03
    model.W_fb_0 = 0.03

    rows = []
    # Loop through
    for r1, r2, r3, r4 in product(R_1, R_2, R_3, R_4):
        # Reset model
        model.reset()

        ## FEED BACK 
        # Init Model Weights
        model.W_r_0 = [0, r1, r2, r3, r4, 0] # Similar results to near far split: [0, 0, 4, 1, 0, 0]
        model.W, W_L = model.init_weights(plot = False, kappas = kappas, return_matrices=True, recurrence = recurrence)
        ## feed back input
        h = vonmises_input(model, k_input, input_layer = 'out')
        h = h / np.linalg.norm(h) / 3
        # simulate
        _, T_fb, W_fb, A_fb = simulate(model, n_timesteps, stim_on, h)

        for i, layer_size in enumerate(model.layer_sizes):
            layer_width_at_stim_off = W_fb[i, stim_on-1]
            layer_width_mean = W_fb[i, :].mean()
            layer_amp_at_stim_off = A_fb[i, stim_on-1]
            layer_amp_mean = A_fb[i, :].mean()
            layer_amp_max = A_fb[i, :].max()

            ydata  = A_fb[i].copy()
            ydata /= ydata.max()
            try:
                l_popt, l_pcov = curve_fit(logistic, xdata=np.arange(0, n_timesteps, 1), ydata=ydata, maxfev=5000)

                l_hat = logistic(np.arange(0, n_timesteps, 1), *l_popt)
                l_err = np.sum(np.square(l_hat - ydata))
            except:
                l_err = np.inf

            row = [r1, r2, r3, r4, i, layer_size, layer_width_at_stim_off, layer_width_mean, layer_amp_at_stim_off, layer_amp_mean, layer_amp_max, l_err, np.std(T_fb)]
            rows.append(row)

    DF = pd.DataFrame(rows, columns = ['r1', 'r2', 'r3', 'r4', 'layer', 'size', 
                                'width_off', 'width_mean', 
                                'amp_off', 'amp_mean', 'amp_max', 
                                'logfit_error', 'timeseries_std'])
    
    DF.to_csv('simulations/memento_recurrence.tsv', sep = '\t')


if __name__ == '__main__':
    main()