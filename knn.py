"""kNN with data (defined in time domain)
    A) channel coefficients
    B) preamble through channel
"""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from scipy import signal as sig

"""Just modify this function."""
def main():
    test_ = [10,50,100,200,400,800,2000,5000,10000]
    params = {
        'k':5,                           # k in kNN
        'db_size':1000,                  # number of examples to use for kNN
        'num_taps':2,                    # number of channel taps [NOTE: channel normalized to "length" 1]
        'preamble_len':10,               # length of the preamble
        'snr':-1,                        # signal to noise ratio; -1 for no noise
        'select':1,                      # 0 for option (A); 1 for option (B) [see description at top of file]
        'num_runs':100,                  # number of random trials to reduce variance in results
        'vary_params':['db_size'],             # parameters to vary [currently varies each parameter individually]
        'vary_ranges':[test_], # range over which to vary
        'fig_dir':"./"                   # where to save figures
        }
    test_many(params)

###################################
#### DON'T MODIFY STUFF BELOW #####
###################################

"""Generate M channels each with N taps. Normalized to 1."""
def gen_rand_channels(m, n):
    channels = np.random.randn(m, n)
    return channels / np.sum(channels**2, axis=1, keepdims=True)**0.5

"""Inverse given the channel coefficients."""
def inverse_from_channels(channels):
    if len(channels.shape) == 1:
        channels = channels.reshape(1,-1)
    temp = -channels
    temp[:,0] = 1. / (channels[:,0] + 1e-3)
    return temp

"""Generate data to send for the BPSK channel."""
def generate_data(n):
    return np.random.choice([-1, 1], n).astype(np.float64)

"""Add Gaussian noise with variance AMPLITUDE to SIGNAL."""
def add_noise(signal, amplitude): 
    return signal + np.sqrt(amplitude) * np.random.randn(len(signal))

"""Apply CHANNEL to CHANNEL_INPUT, with added Gaussian noise such that the
signal to noise ration (SNR) is as input. 

Set SNR to -1 for no noise."""
def apply_channel(channel, channel_input, snr=-1):
    out = sig.convolve(channel_input, channel, mode="full")
    # SNR: signal power is fixed at 1 in our model, so just scale the noise
    if snr > 0:
        out = add_noise(out, 1./snr)
    return out


"""Calculate the number of bit errors between DECODED and SENT.
   Assumes binary channel (values sent are +1 or -1)."""
def bit_error(decoded, sent):
    num_errors = 0
    decoded = np.copy(decoded)
    sent = np.copy(sent)
    for temp in [decoded, sent]:
        temp[temp<0] = 0
        temp[temp>=0] = 1
    return np.sum((np.abs(temp-sent)))/np.size(decoded)

def gen_channel_inv_pairs(m,n):
    c = gen_rand_channels(m,n)
    i = inverse_from_channels(c)
    return c,i

def gen_channel_output_pairs(m,n,p,s):
    c,i = gen_channel_inv_pairs(m,n)
    d = np.array([apply_channel(ci, p, s) for ci in c])
    return d,i,c


def test(k,m,n,l,s,v):
    test_size=m
    if v==1:
        train = gen_channel_inv_pairs(m,n)
        test = gen_channel_inv_pairs(test_size,n)
    elif v==2:
        p = generate_data(l)
        train = gen_channel_output_pairs(m,n,p,s)
        test = gen_channel_output_pairs(test_size,n,p,s)

    kNN = KNeighborsRegressor(k)
    kNN.fit(train[0],train[1])
    predicted = kNN.predict(test[0])

    return np.linalg.norm(predicted - test[1]) / (test_size)
    # bit_error(predicted, test[1])

def test_many(params):
    for i in range(len(params['vary_params'])):
        np.random.seed(0)
        v = params['vary_params'][i]
        orig = params[v]
        y_axis = []
        for j in params['vary_ranges'][i]:
            params[v] = j
            avg_runs = 0.
            for k in range(params['num_runs']):
                avg_runs += test(params['k'], params['db_size'], 
                                 params['num_taps'], params['preamble_len'],
                                 params['snr'], params['select'])
            y_axis.append(avg_runs / params['num_runs'])
        params[v] = orig

        fig, ax = plt.subplots(nrows=1,ncols=1)

        ax.plot(params['vary_ranges'][i], y_axis, label="L2 error of channel inverse")
        ax.set_title("L2 error of channel inverse with kNN: varying "+v)
        ax.set_xlabel(v)
        ax.set_ylabel("L2 error of channel inverse")
        fig.savefig(params['fig_dir']+"plot_"+str(i+1)+".png")
        print(params['fig_dir']+"plot_"+str(i+1)+".png")
        plt.close(fig)

if __name__ == '__main__':
    main()


























