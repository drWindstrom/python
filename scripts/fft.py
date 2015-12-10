# -*- coding: utf-8 -*-
import re
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

def do_fft(signal, freq):
    """Returns the fft."""
    # Create hanning window
    hann = np.hanning(len(signal))
    Y = np.fft.fft(hann*signal)
    N = len(Y)/2+1
    X = np.linspace(0, freq/2, N, endpoint=True)
    return X, Y, N

def load_time_tec(filename):
    """Load convergence data from flower transient simulation."""
    with open(filename) as f:
        f.readline()
        var_names_txt = f.readline()
    # Get variable names
    var_names = re.findall('"(.+?)"', var_names_txt)
    # Load data
    data = np.loadtxt(filename, skiprows=2)
    data_dict = {}
    for i, var_name in enumerate(var_names):
        data_dict[var_name] = data[:, i]
    return data_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('freq')
    args = parser.parse_args()
    conv_data = load_time_tec(args.filename)
    X, Y, N = do_fft(conv_data['cl'], float(args.freq))
    # Plot
    plt.figure('FFT')
    plt.title('FFT')
    plt.plot(X, 2.0*np.abs(Y[:N])/N)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')

# Plot with rescaled x-axis
# Xp = 1.0/X # in seconds
# Xph= Xp/(60.0*60.0) # in hours
# plt.figure(figsize=(15,6))
# plt.plot(Xph, 2.0*np.abs(Y[:N])/N)
# plt.xticks([12, 24, 33, 84, 168])
# plt.xlim(0, 180)
# plt.ylim(0, 1500)
# plt.xlabel('Period ($h$)')
