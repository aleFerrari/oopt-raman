# -*- coding: utf-8 -*-
"""
@Author: Alessio Ferrari
"""
import math as mt
import numpy as np
import pytest
import raman.utilities as ut
from numpy import testing as npt
from collections import namedtuple
from operator import attrgetter


@pytest.mark.parametrize("num_channels", [1, 10])
def test_compute_power_spectrum(num_channels):

    delta_f = 50e9
    pch = 1e-3
    roll_off = 0.1
    symbol_rate = 32e9
    start_f = 191.0e12
    wdm_band = num_channels * delta_f
    stop_f = start_f + wdm_band
    frequency_slice_size = 50e9

    pump_pow = [0.450, 0.400]
    pump_freq = [204.0e12, 206.3e12]
    pump_direction = [-1, -1]
    pump_bandwidth = [1e6, 1e6]
    num_pumps = len(pump_pow)

    spectral_information = namedtuple('SpectralInformation', 'carriers')
    raman_pump_information = namedtuple('SpectralInformation', 'raman_pumps')
    channel = namedtuple('Channel', 'channel_number frequency baud_rate roll_off power')
    power = namedtuple('Power', 'signal nonlinear_interference amplified_spontaneous_emission')
    pump = namedtuple('RamanPump', 'pump_number power frequency propagation_direction pump_bandwidth')

    carriers = tuple(channel(1 + ii, start_f + (delta_f * ii), symbol_rate, roll_off, power(pch, 0, 0))
                                     for ii in range(0, num_channels))
    pumps = tuple(pump(1 + ii, pump_pow[ii], pump_freq[ii], pump_direction[ii], pump_bandwidth[ii])
                                for ii in range(0, num_pumps))
    spec_info = spectral_information(carriers=carriers)
    raman_pump_info = raman_pump_information(raman_pumps=pumps)

    pow_array, f_array, propagation_direction, noise_bandwidth_array = ut.compute_power_spectrum(spec_info, raman_pump_info)

    # Computing expected values for wdm channels
    n_slices = mt.ceil(wdm_band / frequency_slice_size)
    pow_slice = pch * frequency_slice_size / delta_f
    pow_last_slice = (wdm_band / frequency_slice_size) % 1 * pow_slice

    pow_array_test = np.ones(n_slices) * pow_slice
    if pow_last_slice:
        pow_array_test[-1] = pow_last_slice

    f_array_test = np.arange(start_f, stop_f, frequency_slice_size)

    propagation_direction_test = np.ones(num_channels)
    channels_noise_bw_test = np.ones(num_channels)*symbol_rate

    # Computing expected values channels + Raman pumps
    pow_array_test = np.append(pow_array_test, pump_pow)
    f_array_test = np.append(f_array_test, pump_freq)
    propagation_direction_test = np.append(propagation_direction_test, pump_direction)
    noise_bandwidth_array_test = np.append(channels_noise_bw_test, pump_bandwidth)

    npt.assert_allclose(pow_array_test, pow_array, rtol=1e-6)
    npt.assert_allclose(f_array_test, f_array, rtol=1e-6)
    npt.assert_allclose(propagation_direction_test, propagation_direction, rtol=1e-6)
    npt.assert_allclose(noise_bandwidth_array_test, noise_bandwidth_array, rtol=1e-6)


@pytest.mark.parametrize("roll_off", [0, 0.5])
def test_raised_cosine_comb(roll_off):
    # SPECTRAL PARAM
    num_channels = 4
    delta_f = 50e9
    symbol_rate = 32e9
    start_f = 193e12
    pch = 1e-3

    power = namedtuple('Power', 'signal nonlinear_interference amplified_spontaneous_emission')
    channel = namedtuple('Channel', 'channel_number frequency baud_rate roll_off power')
    carriers = tuple(channel(1 + ii, start_f + (delta_f * ii), symbol_rate, roll_off, power(pch, 0, 0))
                     for ii in range(0, num_channels))

    f_eval = np.array([start_f + (ii * delta_f / 2) for ii in range(0, num_channels * 2)])

    psd = ut.raised_cosine_comb(f_eval, *carriers)

    expected_psd = np.array([])
    for ii in range(0, num_channels):
        expected_psd = np.append(expected_psd, [pch / symbol_rate, 0])

    npt.assert_allclose(expected_psd, psd, rtol=1e-5)