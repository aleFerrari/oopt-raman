# -*- coding: utf-8 -*-
"""
@Author: Alessio Ferrari
"""
import numpy as np
from numpy import testing as npt
import pytest
import csv
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

def test_nli():

    # FIBER PARAMETERS

    fiber_information = namedtuple('FiberInformation', 'length attenuation_coefficient raman_coefficient beta2 beta3 gamma')
    attenuation_coefficient_p = namedtuple('Attenuation_coeff','alpha_power')
    attenuation_coefficient_p.alpha_power = np.array([np.log(10) * 0.18895E-3 / 10])
    fiber_information.attenuation_coefficient = attenuation_coefficient_p
    fiber_information.beta2 = -21.27e-27  # s^2/m
    fiber_information.beta3 = beta3 = 0  # s^3/m
    fiber_information.gamma = 1.3e-3  # 1/W/m

    model_parameters = namedtuple('NLIParameters','method frequency_resolution verbose')
    model_parameters.method = 'ggn_integral'
    model_parameters.frequency_resolution = 0.2e9
    model_parameters.verbose = False

    # WDM COMB PARAMETERS
    roll_off = 0.1
    symbol_rate = 32e9

    # SPECTRUM
    spectral_information = namedtuple('SpectralInformation', 'carriers')
    channel = namedtuple('Channel', 'channel_number frequency baud_rate roll_off power')
    power = namedtuple('Power', 'signal nonlinear_interference amplified_spontaneous_emission')

    # COMPARISON WITH MATHLAB RESULTS

    csv_files_dir = './resources/'
    f_axis = (1E+12) * np.loadtxt(open(csv_files_dir + 'f_axis.csv', 'rb'), delimiter=',')
    z_array = (1E+3) * np.loadtxt(open(csv_files_dir + 'z_array.csv', 'rb'), delimiter=',')
    rho = np.loadtxt(open(csv_files_dir + 'raman_profile.csv'), delimiter=',')
    A = np.exp((-attenuation_coefficient_p.alpha_power / 2) * z_array)
    for i in range(len(rho)):
        rho[i] = np.multiply(rho[i], A)
    f_channel = (1E+12) * np.loadtxt(open(csv_files_dir + 'f_channel.csv', 'rb'), delimiter=',')
    f_channel = f_channel[range(11)]
    l = len(f_channel)
    cut_index = [*range(1,l+1)]
    pch = 0.50119E-03 * np.ones(l)
    channel_numbers = range(l)
    carriers = tuple(channel(i + 1, f_channel[i], symbol_rate, roll_off, power(pch[i], 0, 0)) for i in channel_numbers)
    spectrum = spectral_information(carriers=carriers)
    raman_solver = namedtuple('RamanSolver', 'raman_bvp_solution spectral_information')
    raman_solver.spectral_information = spectrum
    raman_bvp_solution = namedtuple('raman_bvp_solution', ' rho z frequency ')
    raman_bvp_solution = raman_bvp_solution(rho=rho, z=z_array, frequency=f_axis)
    raman_solver = raman_solver(raman_bvp_solution=raman_bvp_solution, spectral_information=spectrum)

    nlint = nli.NLI(fiber_information=fiber_information)
    nlint.srs_profile = raman_solver
    nlint.model_parameters = model_parameters
    carriers_nli = [nlint.compute_nli(carrier, *carriers)
                    for carrier in carriers if (carrier.channel_number in cut_index)]

    # PLOT RESULTS
    p_cut = [carrier.power.signal for carrier in sorted(spectrum.carriers, key=attrgetter('frequency')) if
             (carrier.channel_number in cut_index)]
    f_cut = [carrier.frequency for carrier in sorted(spectrum.carriers, key=attrgetter('frequency')) if
             (carrier.channel_number in cut_index)]

    rho_end = interp1d(raman_solver.raman_bvp_solution.frequency, raman_solver.raman_bvp_solution.rho[:, -1])
    p_cut = np.array(p_cut) * (rho_end(f_cut)) ** 2

    snr_nl = 10*np.log10(p_cut / carriers_nli)
    print(snr_nl)
    OSNR_NL = [34.87001733, 34.58858743, 34.46470382, 34.38424991, 34.32143473, 34.28745844, 34.26041349,
               34.25713172, 34.27668589, 34.35232055, 34.60162964]
# [36.569354, 35.9477  , 35.639568, 35.442189, 35.309405, 35.233355,35.202252, 35.219708, 35.305059, 35.512505, 36.076057]
    npt.assert_allclose(snr_nl, OSNR_NL, rtol=1E-6)



















