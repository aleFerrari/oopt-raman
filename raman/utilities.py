# -*- coding: utf-8 -*-
import numpy as np
from operator import attrgetter


def compute_power_spectrum(spectral_information, raman_pump_information):
    """
    Rearrangement of spectral and Raman pump information to make them compatible with Raman solver
    :param spectral_information: a namedtuple describing the transmitted channels
    :param raman_pump_information: a namedtuple describing the Raman pumps
    :return:
    """

    # Signal power spectrum
    pow_array = np.array([])
    f_array = np.array([])
    noise_bandwidth_array = np.array([])
    for carrier in sorted(spectral_information.carriers, key=attrgetter('frequency')):
        f_array = np.append(f_array, carrier.frequency)
        pow_array = np.append(pow_array, carrier.power.signal)
        noise_bandwidth_array = np.append(noise_bandwidth_array, carrier.baud_rate)

    propagation_direction = np.ones(len(f_array))

    # Raman pump power spectrum
    for pump in raman_pump_information.raman_pumps:
        pow_array = np.append(pow_array, pump.power)
        f_array = np.append(f_array, pump.frequency)
        propagation_direction = np.append(propagation_direction, pump.propagation_direction)
        noise_bandwidth_array = np.append(noise_bandwidth_array, pump.pump_bandwidth)

    # Final sorting
    ind = np.argsort(f_array)
    f_array = f_array[ind]
    pow_array = pow_array[ind]
    propagation_direction = propagation_direction[ind]

    return pow_array, f_array, propagation_direction, noise_bandwidth_array


def raised_cosine_comb(f, *carriers):
    """ Returns an array storing the PSD of a WDM comb of raised cosine shaped
    channels at the input frequencies defined in array f

    :param f: numpy array of frequencies in Hz
    :param carriers: namedtuple describing the WDM comb
    :return: PSD of the WDM comb evaluated over f
    """
    psd = np.zeros(np.shape(f))
    for carrier in carriers:
        f_nch = carrier.frequency
        g_ch = carrier.power.signal / carrier.baud_rate
        ts = 1 / carrier.baud_rate
        passband = (1 - carrier.roll_off) / (2 / carrier.baud_rate)
        stopband = (1 + carrier.roll_off) / (2 / carrier.baud_rate)
        ff = np.abs(f - f_nch)
        tf = ff - passband
        if carrier.roll_off == 0:
            psd = np.where(tf <= 0, g_ch, 0.) + psd
        else:
            psd = g_ch * (np.where(tf <= 0, 1., 0.) + 1 / 2 * (1 + np.cos(np.pi * ts / carrier.roll_off * tf)) *
                          np.where(tf > 0, 1., 0.) * np.where(np.abs(ff) <= stopband, 1., 0.)) + psd

    return psd
