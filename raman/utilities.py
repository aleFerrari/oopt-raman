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
    for carrier in sorted(spectral_information.carriers, key=attrgetter('frequency')):
        f_array = np.append(f_array, carrier.frequency)
        pow_array = np.append(pow_array, carrier.power.signal)

    propagation_direction = np.ones(len(f_array))

    # Raman pump power spectrum
    for pump in raman_pump_information.raman_pumps:
        pow_array = np.append(pow_array, pump.power)
        f_array = np.append(f_array, pump.frequency)
        propagation_direction = np.append(propagation_direction, pump.propagation_direction)

    # Final sorting
    ind = np.argsort(f_array)
    f_array = f_array[ind]
    pow_array = pow_array[ind]
    propagation_direction = propagation_direction[ind]

    return pow_array, f_array, propagation_direction