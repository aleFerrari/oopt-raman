# -*- coding: utf-8 -*-
"""
@Author: Alessio Ferrari
"""
import numpy as np
from numpy import testing as npt
import pytest
from raman import nli
from collections import namedtuple


def test_nli_fiber_information():
    fiber_length = np.array([80e3])
    attenuation_coefficient_p = np.array([0.046e-3])
    frequency_attenuation = np.array([193.5e12])
    gamma = 1.27e-3     # 1/W/m
    beta2 = 21.27e-27   # s^2/m
    beta3 = 3.90986364e-3   # s^3/m

    fiber_information = namedtuple('FiberInformation', 'length attenuation_coefficient beta2 beta3 gamma')
    attenuation_coefficient = namedtuple('AttenuationCoefficient', 'alpha_power frequency')

    att_coeff = attenuation_coefficient(alpha_power=attenuation_coefficient_p, frequency=frequency_attenuation)
    fib_info = fiber_information(length=fiber_length, attenuation_coefficient=att_coeff,
                                        gamma=gamma, beta2=beta2, beta3=beta3)

    nli_model = nli.NLI(fiber_information=fib_info)

    assert nli_model.fiber_information.length == fiber_length
    assert nli_model.fiber_information.gamma == gamma
    assert nli_model.fiber_information.beta2 == beta2
    assert nli_model.fiber_information.beta3 == beta3
    assert nli_model.fiber_information.attenuation_coefficient.alpha_power == attenuation_coefficient_p
    assert nli_model.fiber_information.attenuation_coefficient.frequency == frequency_attenuation


def test_nli_parameters():
    frequency_resolution = 2e9
    verbose = 1

    nli_parameters = namedtuple('NLIParameters', 'frequency_resolution verbose')

    nli_params = nli_parameters(frequency_resolution=frequency_resolution, verbose=verbose)

    nli_model = nli.NLI()
    nli_model.nli_parameters = nli_params

    assert nli_model.nli_parameters.frequency_resolution == frequency_resolution
    assert nli_model.nli_parameters.verbose == verbose

@pytest.mark.parametrize("delta_beta, x_talk", [(3,1E-3),(1E+3,-1E-3),(1E-3,0),(1E-3,-1E-3),(0,1E-3),(1E-3,-1E-4)])
def test_nli_fwm_efficiency(delta_beta,x_talk):

    NLI = nli.NLI

    z = np.arange(0,81E+3,1E+3)
    delta_rho = np.array([np.exp(x_talk*z)])
    alpha0 = 1E-20*np.log(10)*0.18895E-3/10

    def exp_integral(z,w):
        integral = (np.exp(w*z[-1])-np.exp(w*z[0]))/w
        return integral

    expected = np.array([np.abs(exp_integral(z,(1j*delta_beta-alpha0+x_talk)))**2])
    calculed = NLI._fwm_efficiency(delta_beta, delta_rho, z, alpha0)

    npt.assert_allclose(calculed,expected,rtol=1E-5)

    delta_rho = np.array([x_talk * z])

    def linear_integral(z,w,x_talk):
        total = x_talk*(z[-1]*np.exp(w*z[-1])-z[0]*np.exp(w*z[0]))/w
        derivate = x_talk*(np.exp(w*z[-1])-np.exp(w*z[0]))/(w**2)

        return total + derivate

    expected = np.array([np.abs(linear_integral(z, (1j * delta_beta - alpha0),x_talk)) ** 2])
    calculed = NLI._fwm_efficiency(delta_beta, delta_rho, z, alpha0)

    #npt.assert_allclose(calculed, expected, rtol=1E-3)










