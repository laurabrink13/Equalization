"""kNN with data (defined in time domain)
    A) channel coefficients
    B) preamble through channel

    I think it makes most sense just to import this file (e.g., import knn).
    Then call knn.test_many(params) (use text in main() as a guide for how
    to use it).

    NOTE: I am thresholding the channel taps. I found that for option (A) above,
    where you threshold does not matter, but for option (B) the effect is much
    stronger and you need to choose a higher threshold value (depending on noise
    level).

    Also note that I am normalizing the channel (e.g., l2 norm of channel taps
    is set to 1). Thresholding occurs before normalization.
"""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import matplotlib.pyplot as plt
from scipy import signal as sig

"""Just modify this function."""
def main():
    test_ = [10, 50, 100, 500, 1000, 5000]
    params = {
        'k':9,                     # k in kNN
        'db_size':100,             # number of examples to use for kNN (training set size)
        'num_taps':20,             # number of channel taps [NOTE: channel normalized to "length" 1]
        'preamble_len':10,         # length of the preamble
        'snr':1e0,                 # signal to noise ratio; -1 for no noise (only matters for (B))
        'select':2,                # 0 for option (A); 1 for option (B) [see description at top of file]
        'num_runs':100,            # number of random trials to reduce variance in results
        'vary_params':['db_size'], # parameters to vary [currently varies each parameter individually]
        'vary_ranges':[test_],     # range over which to vary
        'fig_dir':"../"            # where to save figures
        }
    # RESULT is a list
    # each entry in the list (e.g., result[0]) is a list of the avg l2 distance
    #   from predicted inverse to actual inverse while varying the ith parameter
    #   in params[`vary_params`]
    # for example, using the above settins as an example,
    #   result[0][0] contains the avg. l2 distance with the above settins and
    #   params['db_size'] set to 10.
    result = test_many(params,plot_here=True)

###################################
#### DON'T MODIFY STUFF BELOW #####
###################################

"""Generate M channels each with N taps. Normalized to 1."""
def gen_rand_channels(m, n):
    channels = np.random.randn(m, n)
    # thresholding smallest tap magnitude of channel -- doesn't matter where
    # if you do not outlier test channels give the most error and skew results
    # for (A) -- does not matter where you threshold
    # for (B) -- setting much lower than 5e-1 makes it difficult to see
    #            expected results
    if False:
        thresh = 2e-1
        channels[np.abs(channels)<thresh] = thresh*np.sign(channels[np.abs(channels)<thresh])
    # could also just assert first tap is always 1 (don't do above then)
    # note that it won't actually be 1 because we normalize channel after;
    # this also works in giving expected results of kNN
    # this method works more consistently..
    #
    # I think it makes sense since it basically just ensures inverting is not
    # ill-conditioned
    else:   
        channels[:,0] = 1*np.sign(channels[:,0])
    return channels / np.sum(channels**2, axis=1, keepdims=True)**0.5

"""Inverse given the channel coefficients.
   e.g., for channel coeffs [a0, a1, a2, ..., an], generates
                            [1/a0, -a1, -a2, ..., -an].
   Operates on a numpy array, each row being a channel."""
def inverse_from_channels(channels):
    if len(channels.shape) == 1:
        channels = channels.reshape(1,-1)
    temp = -channels
    temp[:,0] = 1. / (channels[:,0])
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

"""Generates M random channels each with N taps and their 'inverse's."""
def gen_channel_inv_pairs(m,n):
    c = gen_rand_channels(m,n)
    i = inverse_from_channels(c)
    return c,i

"""Generates a M random channels with N taps, the channel 'inverse's, and a
   the result of sending preamble P through the channels with noise snr S."""
def gen_channel_output_pairs(m,n,p,s):
    c,i = gen_channel_inv_pairs(m,n)
    d = np.array([apply_channel(ci, p, s) for ci in c])
    return d,i,c

"""Uses kNN regression to predict channel inverses.
   k -- k in kNN
   m -- number of channels to train on
   n -- number of taps in each channel
   l -- length of preamble
   s -- noise level when sending preamble through channel
   v -- 1 for option (A), 2 for option (B) (see top of page)
"""
def test(k,m,n,l,s,v):
    test_size = 100
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
    return np.linalg.norm(predicted - test[1]) / (len(test[0]))

"""Iterates through changing parameters and generates plots."""
def test_many(params, plot_here=False):
    all_results = []
    for i in range(len(params['vary_params'])):
        np.random.seed(10)
        v = params['vary_params'][i]
        print("Varying parameter: " + v)
        orig = params[v]
        y_axis = []
        for j in params['vary_ranges'][i]:
            params[v] = j
            avg_runs = 0.
            for k in range(params['num_runs']):
                avg_runs += test(params['k'], params['db_size'], 
                                 params['num_taps'], params['preamble_len'],
                                 params['snr'], params['select'])
            avg_runs = avg_runs / params['num_runs']
            y_axis.append(avg_runs)
            print(v+" = " + str(j) + ": " + str(avg_runs))
        params[v] = orig

        # generate the plot
        if plot_here:
            fig, ax = plt.subplots(nrows=1,ncols=1)
            ax.plot(params['vary_ranges'][i], y_axis, label="L2 error of channel inverse")
            ax.set_title("L2 error of channel inverse with kNN: varying "+v)
            ax.set_xlabel(v)
            ax.set_ylabel("L2 error of channel inverse")
            fig.savefig(params['fig_dir']+"plot_"+str(i+1)+".png")
            plt.close(fig)
        # mostly for debugging purposes, set to True
        if True:
            print(params['fig_dir']+"plot_"+str(i+1)+".png")

        all_results.append(y_axis)
    return all_results


# for stand-alone script
if __name__ == '__main__':
    main()


























