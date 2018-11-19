# -*- coding: utf-8 -*-
"""
@Author: Alessio Ferrari
"""
import numpy as np
from numpy import testing as npt
import pytest
from raman import nli
from collections import namedtuple
import raman.raman as rm
from operator import attrgetter
from scipy.interpolate import interp1d
import raman.utilities as ut
import progressbar

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

@pytest.mark.parametrize("delta_beta, x_talk, delta_rho_fun", [(3,[1E-3, 1],"linear"),(1E+3,[-1E-3,0],"linear"),(1E-3,[-1E-3,0],"exponential")])#,(0,1E-3,"exponential"),(1E-3,-1E-4,"exponential")])
def test_nli_fwm_efficiency(delta_beta,x_talk,delta_rho_fun):

    NLI = nli.NLI

    z = np.arange(0,81E+3,1)
    alpha0 = 1E-20*np.log(10)*0.18895E-3/10
    w = 1j*delta_beta-alpha0

    def evaluate(vec):
        return vec[-1]-vec[0]

    delta_rho=0
    der_delta_rho=0
    expected=0
    exponential=np.exp(w*z)
    int_exponential = exponential/w



    if delta_rho_fun == "linear":
        delta_rho = x_talk[0]*z+x_talk[1]*np.ones(len(z))
        der_delta_rho = x_talk[0]*np.ones(len(z))
        total = int_exponential*delta_rho
        other = - int_exponential*der_delta_rho/w
        expected = np.array([np.abs(evaluate(total+other))**2])

    elif delta_rho_fun == "exponential":
        delta_rho = np.exp(x_talk[0]*z)
        exponential = np.exp((w+x_talk[0])*z)
        int_exponential = exponential / (w+x_talk[0])
        expected=np.array([np.abs(evaluate(int_exponential))**2])

    calculed = NLI._fwm_efficiency(delta_beta, np.array([delta_rho]), z, alpha0)

    npt.assert_allclose(calculed,expected,rtol=1E-5)





















