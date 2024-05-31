import numpy as np
import scipy.stats as stats
from scipy.signal import peak_widths, find_peaks
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import itertools
from tqdm import tqdm
import pandas as pd

# The model 
class Memento:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.layers = [np.zeros([m, 1]) for m in self.layer_sizes]

        self.r = np.concatenate(self.layers)
        self.W = np.zeros([len(self.r), len(self.r)])

        self.in_idx = np.arange(0, layer_sizes[0])
        self.out_idx = np.arange(len(self.r) - layer_sizes[-1], len(self.r))
        self.vth = np.zeros_like(self.r)

        self.layer_indices = self.generate_layer_indices()

        self.W_ff_0 = 0.00
        self.W_ff_1 = 0.00
        self.W_fb_0 = 0.00
        self.W_fb_1 = 0.00

        self.W_r_0 = np.zeros([len(self.layer_sizes)])
        self.W_r_1 = np.zeros([len(self.layer_sizes)])

        self.dt = 1
        self.tau = 10   

    
    def generate_layer_indices(self):
        start = 0
        layer_indices = []
        for l in self.layer_sizes:
            stop = start + l
            interval = [start, stop] 
            indices = np.arange(interval[0], interval[1])
            layer_indices.append(indices)
            start += l
        return layer_indices
        

    
    def init_weights(self, kappas = [], plot = False, return_matrices = False, recurrence = False, norm = False):
        W = np.zeros_like(self.W)
        W_L = []

        
        if not kappas:
            self.kappas = [1 for i in range(len(self.layer_sizes))]
        else:
            self.kappas = kappas

        # FF and FB connection weights
        for i in range(len(self.layer_sizes) - 1):
            rstart = int(np.sum(self.layer_sizes[:(i+1)]))
            rstop = rstart + self.layer_sizes[i+1]

            cstart = int(np.sum(self.layer_sizes[:i]))
            cstop = cstart + self.layer_sizes[i]

            shape = (self.layer_sizes[i+1], self.layer_sizes[i])

            W_l = np.zeros(shape)
            x = np.linspace(0, 2*np.pi, shape[1])

            for ii in range(shape[0]):
                loc = ((ii/shape[0]) * 2*np.pi)
                rf = stats.vonmises.pdf(x, self.kappas[i], loc)
                if norm: rf /= np.linalg.norm(rf)
                W_l[ii, :] = rf

            
            W_L.append(W_l)
            
            # FF Weights
            W[rstart:rstop, cstart:cstop] = W_l  * self.W_ff_0 + self.W_ff_1
            # FB weights
            W[cstart:cstop, rstart:rstop] = W_l.T  * self.W_fb_0 + self.W_fb_1

        if recurrence == 'vonmises':
            start = 0
            k1, k2, s1, s2 = 10, 1, 1, 1
            for i in range(len(self.layer_sizes)):
                stop = int(np.sum(self.layer_sizes[:(i+1)]))
                
                shape = [self.layer_sizes[i], self.layer_sizes[i]]
                W_l = np.zeros(shape)
                x = np.linspace(0, 2*np.pi, shape[1])

                for ii in range(shape[0]):
                    loc = ((ii/shape[0]) * 2*np.pi)
                    rf = stats.vonmises.pdf(x, k1, loc, scale = s1) - stats.vonmises.pdf(x, k2, loc, scale = s2) 
                    rf /= np.linalg.norm(rf)
                    W_l[ii, :] = rf * 1

                W[start:stop, start:stop] = W_l
                W_L.append(W_l)

                start = int(np.sum(self.layer_sizes[:(i+1)]))

        if recurrence == 'cosine':
            start = 0
            for i in range(len(self.layer_sizes)):
                stop = int(np.sum(self.layer_sizes[:(i+1)]))

                W_l = self.cos_weights(layer = i, W_0 = self.W_r_0[i], W_1 = self.W_r_1[i])

                W[start:stop, start:stop] = W_l
                W_L.append(W_l)

                start = int(np.sum(self.layer_sizes[:(i+1)]))

        if recurrence == 'cosine2':
            start = 0
            for i in range(len(self.layer_sizes)):
                stop = int(np.sum(self.layer_sizes[:(i+1)]))

                W_l = self.cos_weights2(layer = i, W_0 = self.W_r_0[i])

                W[start:stop, start:stop] = W_l
                W_L.append(W_l)

                start = int(np.sum(self.layer_sizes[:(i+1)]))


        if plot:
            plt.imshow(W, vmin = -np.max(np.abs(W)), vmax = np.max(np.abs(W)), cmap = 'bwr')
            plt.title("Model Weights")
            plt.colorbar()

        if return_matrices:
            return W, W_L


        W /= np.linalg.norm(W)
        return W
    

    def step(self, r):    
        self.r = self.W @ r
        self.r /= np.linalg.norm(self.r)
        # self.r /= self.r.max()
        
        return self.r
    

    def euler(self, h):
        r = np.zeros_like(h)
        y = self.W @ self.r + h - self.vth
        y[y < 0] = 0

        r[:] = (1 - (self.dt/self.tau)) * self.r + (self.dt/self.tau) * y
        
        return r
    

    def W_theta_old(self, theta, W_0, W_1):
        return W_1 + 2*W_0 * np.cos(theta)
    
    def W_theta(self, theta, W_0):
        return W_0 * np.cos(theta)


    def cos_weights(self, layer, W_0 = 0.3, W_1 = 1.5):
        delta = 2*np.pi/self.layer_sizes[0]
        grid_positions = np.arange(0, self.layer_sizes[layer])

        theta = np.linspace(0, 2*np.pi, self.layer_sizes[layer])

        W = np.zeros([theta.size, theta.size])

        # Set weights
        for i, j, in itertools.product(grid_positions, grid_positions):
            W[i, j] = self.W_theta_old(theta[i] - theta[j],  W_0, W_1) * (delta / (2*np.pi))

        return W
    

    def cos_weights2(self, layer, W_0 = 0.3):
        delta = 2*np.pi/self.layer_sizes[0]
        grid_positions = np.arange(0, self.layer_sizes[layer])

        theta = np.linspace(0, 2*np.pi, self.layer_sizes[layer])

        W = np.zeros([theta.size, theta.size])

        # Set weights
        for i, j, in itertools.product(grid_positions, grid_positions):
            W[i, j] = self.W_theta(theta[i] - theta[j],  W_0) * (delta / (2*np.pi))

        W -= W.max()

        return W
    

    def reset(self):
        self.layers = [np.zeros([m, 1]) for m in self.layer_sizes]

        self.r = np.concatenate(self.layers)

    
    def compute_widths(self):
        start = 0
        widths = np.zeros(len(self.layer_sizes))

        for i, layer in enumerate(self.layer_sizes):
            stop = start + layer
            response = self.r[start:stop].squeeze()

            peak, _ = find_peaks(response)

            width, _, _, _ = peak_widths(response, peak)

            if len(width) != 1:
                width = 0
            widths[i] = (width/layer) * 360
            
            start += self.layer_sizes[i]

        return widths
    

    def compute_amplitudes(self):
        start = 0
        amps = np.zeros(len(self.layer_sizes))

        for i, layer in enumerate(self.layer_sizes):
            stop = start + layer
            response = self.r[start:stop].squeeze()
            
            amps[i] = response.max()

            start += self.layer_sizes[i]

        return amps


    def plot(self, timecourse, h, stim_on = 0):
        # Plotting
        fig, axs = plt.subplots(2, 2, figsize = [18, 12])

        # Timecourse
        ax = axs[0, 1]
        fig1 = ax.imshow(timecourse, aspect = 'auto', interpolation='none', cmap = 'RdBu_r')
        # cbar = fig.colorbar(fig1)
        # cbar.minorticks_on()
        ax.set_title("Response Timecourse")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Nodes")
        pos = 0
        for i, layer in enumerate(self.layer_sizes):
            ax.text(0, 1-(pos/self.r.shape[0]), "Layer %d" % i, horizontalalignment='left',
                        verticalalignment='top', transform=ax.transAxes, color = 'white')
            pos += layer
            if i != len(self.layer_sizes)-1:
                ax.axhline(pos, color = 'black')
            
            
        # Amplitude
        ax = axs[1, 0]
        start = 0
        for i in range(len(self.layer_sizes)):
            if i+1:
                midpoint = int(start + self.layer_sizes[i]/2)
                timeseries = timecourse[midpoint, :]
                ax.plot(timeseries, label = 'Layer %d' % i)
            start += self.layer_sizes[i]

        ax.hlines(h.max(), xmin = 0, xmax = timecourse.shape[1], color = 'gray', linestyle = "--", label = 'Input Amplitude')
        if stim_on:
            ax.axvline(self, ymin = 0, ymax = 1, color = 'red', label = 'Stim Offset', alpha = 0.5)
        ax.legend()
        ax.set_title("Response Amplitude")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Amplitude at Center (Arbitrary Units)")

        # Compute widths in each layer, plot over time
        ax = axs[1, 1]

        start = 0
        Widths = np.zeros([len(self.layer_sizes), timecourse.shape[1]])

        for i, layer in enumerate(self.layer_sizes):
            stop = start + layer
            layer_timecourse = timecourse[start:stop, :]

            for t, response in enumerate(layer_timecourse.T):
                
                peak, _ = find_peaks(response)
                width, _, _, _ = peak_widths(response, peak)
                if len(width) != 1:
                    width = 0
                Widths[i, t] = (width/layer) * 360
            start += self.layer_sizes[i]
            ax.plot(Widths[i, :], label = 'Layer %d' % i)

        h_peak, _ = find_peaks(h.squeeze())
        h_width, _, _, _ = peak_widths(h.squeeze(), h_peak)
        h_width = h_width/h.shape[0] * 360

        ax.hlines(h_width, xmin = 0, xmax = timecourse.shape[1], color = 'gray', linestyle = "--", label = 'Input Width')
        if stim_on:
            ax.axvline(stim_on, ymin = 0, ymax = 1, color = 'red', alpha = 0.5, label = 'Stim Offset')
        ax.legend()
        ax.set_title("FWHM")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Width (in deg)")

        ax = axs[0, 0]
        # eigvals = np.linalg.eigvals(self.W)
        # # extract real part 
        # x = [ele.real for ele in eigvals] 
        # # extract imaginary part 
        # y = [ele.imag for ele in eigvals]
        # ax.scatter(x, y)
        # circ = plt.Circle((0, 0), radius=1, edgecolor='b', facecolor='None')
        # circ2 = plt.Circle((0, 0), radius=0.5, edgecolor='b', facecolor='None')
        # circ3 = plt.Circle((0, 0), radius=1.5, edgecolor='b', facecolor='None')
        # ax.add_patch(circ)
        # ax.add_patch(circ2)
        # ax.add_patch(circ3)
        # ax.set_title("Eigenvalues")
        # ax.set_xlabel("Real")
        # ax.set_ylabel("Imag")

        v = np.abs(np.asarray([self.W.min(), self.W.max()])).max()
        ax.imshow(self.W, aspect = 'auto', interpolation = 'none', vmin = -v, vmax = v, cmap = 'RdBu_r')    
        ax.set_aspect('equal', 'box')
        ax.set_title('Weight Matrix')
        plt.show()
            
# Helper functions      
def simulate_both(model, n_timesteps, stim_on, layer_sizes, k_input):
    # Each Layer's Center locations for computing peak tuning widths
    peaks = []
    pos = 0
    for i in range(len(model.layer_sizes)):
        peak  = pos + int(model.layer_sizes[i]/2) 
        peaks.append(peak)
        pos += model.layer_sizes[i]

    # Feedforward
    x_pos = 180
    theta = np.linspace(0, 2*np.pi, layer_sizes[0])
    h = stats.vonmises.pdf(theta, k_input, np.radians(x_pos), 1)
    # h/= h.max()
    h = np.expand_dims(h, 1)

    timecourse_ff = np.zeros([model.r.shape[0], n_timesteps])
    H_ff = np.zeros_like(timecourse_ff)
    Widths_ff = np.zeros([len(model.layer_sizes), n_timesteps])
    Amps_ff = np.zeros([len(model.layer_sizes), n_timesteps])

    for t in range(n_timesteps):
        x_in = np.zeros_like(model.r)
        if t < stim_on:
            x_in[model.in_idx] = h
            model.r = model.euler(h = x_in)
        else:
            model.r = model.euler(h = x_in)

        timecourse_ff[:, t] = model.r.squeeze()
        H_ff[:, t] = x_in.squeeze()
        Widths_ff[:, t] = model.compute_widths()
        Amps_ff[:, t] = model.compute_amplitudes()

    # Feedback
    h = model.r[model.out_idx]
    # plt.plot(h)
    model.reset()

    theta = np.linspace(0, 2*np.pi, layer_sizes[-1])
    # h = stats.vonmises.pdf(theta, k_input, np.radians(x_pos), 1)
    # h = np.expand_dims(h, 1)

    timecourse_fb = np.zeros([model.r.shape[0], n_timesteps])
    H_fb = np.zeros_like(timecourse_fb)
    Widths_fb = np.zeros([len(model.layer_sizes), n_timesteps])
    Amps_fb = np.zeros([len(model.layer_sizes), n_timesteps])

    for t in range(n_timesteps):
        x_in = np.zeros_like(model.r)
        if t < stim_on:
            x_in[model.out_idx] = h
            model.r = model.euler(h = x_in)
        else:
            model.r = model.euler(h = x_in)

        timecourse_fb[:, t] = model.r.squeeze()
        H_fb[:, t] = x_in.squeeze()
        Amps_fb[:, t] = model.compute_amplitudes()
        Widths_fb[:, t] = model.compute_widths()

    FF= {
        'timecourse': timecourse_ff,
        'input': H_ff,
        'amp': Amps_ff,
        'widths': Widths_ff
    }
    FB= {
        'timecourse': timecourse_fb,
        'input': H_fb,
        'amp': Amps_fb,
        'widths': Widths_fb
    }

    return FF, FB


def simulate(model, n_timesteps, stim_on, h):
    # Initialize empty arrays.
    timecourse = np.zeros([model.r.shape[0], n_timesteps])
    H = np.zeros_like(timecourse)
    Widths = np.zeros([len(model.layer_sizes), n_timesteps])
    Amps = np.zeros([len(model.layer_sizes), n_timesteps])
    h_0 = np.zeros_like(model.r)

    # Simulate
    for t in range(n_timesteps):
        if t < stim_on:
            model.r = model.euler(h = h)
            H[:, t] = h.squeeze()
        else:
            model.r = model.euler(h = h_0)
            H[:, t] = h_0.squeeze()

        timecourse[:, t] = model.r.squeeze()
        Widths[:, t] = model.compute_widths()
        Amps[:, t] = model.compute_amplitudes()

    return H, timecourse, Widths, Amps
    

def vonmises_input(model, k_input, input_layer = 'in', x_pos = 180):
    # Feedforward

    h = np.zeros_like(model.r)

    if input_layer == 'in':
        theta = np.linspace(0, 2*np.pi, model.layer_sizes[0])
        x_in = stats.vonmises.pdf(theta, k_input, np.radians(x_pos), 1)
        x_in = np.expand_dims(x_in, 1)
        h[model.in_idx] = x_in

    elif input_layer == 'out':
        theta = np.linspace(0, 2*np.pi, model.layer_sizes[-1])
        x_in = stats.vonmises.pdf(theta, k_input, np.radians(x_pos), 1)
        x_in = np.expand_dims(x_in, 1)
        h[model.out_idx] = x_in

    return h


def logistic(x, k, x0, c):
    y = (1 / (1 + np.exp(-k * (x - x0)))) + np.log(c)
    return y


def exponential(x, a, b):
    y = (a + 2**x)/b
    return y

