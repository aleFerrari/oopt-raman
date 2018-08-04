# -*- coding: utf-8 -*-
"""
@Author: Alessio Ferrari
"""
import numpy as np
import pytest
import raman.raman as rm
from collections import namedtuple
from numpy import testing as npt


@pytest.mark.parametrize("power_spectrum, prop_direction", [([1e-3], [+1]), ([1e-3, 0.4], [+1, -1]), ([1e-3, 1e-3, 0.4], [+1, +1, -1])])
def test_fiber_configuration(power_spectrum, prop_direction):

    fiber_length = np.array([80e3])
    attenuation_coefficient_p = np.array([0.046e-3])
    frequency_attenuation = np.array([193.5e12])
    cr = np.array([0.39])
    frequency_cr = np.array([193.0e12])

    fiber_information = namedtuple('FiberInformation', 'fiber_length attenuation_coefficient raman_coefficient')
    attenuation_coefficient = namedtuple('AttenuationCoefficient', 'alpha_power frequency')
    raman_coefficient = namedtuple('RamanCoefficient', 'cr frequency')

    att_coeff = attenuation_coefficient(alpha_power=attenuation_coefficient_p, frequency=frequency_attenuation)
    raman_coeff = raman_coefficient(cr=cr, frequency=frequency_cr)
    fib_info = fiber_information(fiber_length=fiber_length, attenuation_coefficient=att_coeff, raman_coefficient=raman_coeff)

    raman_solver = rm.RamanSolver(fiber_information=fib_info)

    assert raman_solver.fiber_information.fiber_length == fiber_length
    assert raman_solver.fiber_information.attenuation_coefficient.alpha_power == attenuation_coefficient_p
    assert raman_solver.fiber_information.attenuation_coefficient.frequency == frequency_attenuation
    assert raman_solver.fiber_information.raman_coefficient.cr == cr
    assert raman_solver.fiber_information.raman_coefficient.frequency == frequency_cr

    z = np.arange(0, 90e3, 10e3)
    power_spectrum = np.array(power_spectrum)
    prop_direction = np.array(prop_direction)
    alphap = attenuation_coefficient_p * np.ones(power_spectrum.size)

    actual_power = 10 * np.log10(raman_solver._initial_guess_raman(z, power_spectrum, alphap, prop_direction))

    for index, power in enumerate(power_spectrum):
        desired_power = 10*np.log10(np.exp(-alphap[index] * z[::prop_direction[index]]) * power_spectrum[index])
        npt.assert_allclose(actual_power[index], desired_power, rtol=1e-5)
