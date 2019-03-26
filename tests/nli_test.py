# -*- coding: utf-8 -*-
"""
@Author: Alessio Ferrari
"""
import numpy as np
from numpy import testing as npt
import pytest
from raman import nli
from raman import raman as rm
from collections import namedtuple
from scipy.interpolate import interp1d


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
    fib_info = fiber_information(length=fiber_length, attenuation_coefficient=att_coeff, gamma=gamma, beta2=beta2, beta3=beta3)

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


@pytest.mark.parametrize("delta_beta, x_talk, delta_rho_fun", [(3, [1E-3, 1], "linear"), (1E+3, [-1E-3, 0], "linear"),
                                                               (1E-3, [-1E-3, 0], "exponential")])  #,(0,1E-3,"exponential"),(1E-3,-1E-4,"exponential")])
def test_nli_fwm_efficiency(delta_beta, x_talk, delta_rho_fun):

    NLI = nli.NLI

    z = np.arange(0,81E+3,1)
    alpha0 = 1E-20*np.log(10)*0.18895E-3/10
    w = 1j*delta_beta-alpha0

    def evaluate(vec):
        return vec[-1]-vec[0]

    delta_rho=0
    # der_delta_rho=0
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

    npt.assert_allclose(calculed, expected, rtol=1E-5)

def test_disaggregated_gn_vs_ggn():
        # FIBER PARAMETERS
        cr = 0
        frequency_cr = 0

        fiber_length = np.array([75e3])
        attenuation_coefficient_p = np.array([0.046e-3])
        frequency_attenuation = np.array([193.5e12])

        gamma = 1.27e-3  # 1/W/m
        beta2 = 21.27e-27  # s^2/m
        beta3 = 0.0344e-39  # s^3/m
        f_ref_beta = 19.35e12  # Hz

        # WDM COMB PARAMETERS
        num_channels = 11
        delta_f = 50e9
        pch = 0.5e-3

        cut_list = range(5, num_channels, 5)
        cut_list = [45]

        roll_off = 0.1
        symbol_rate = 32e9
        start_f = 191.0e12

        # ODE SOLVER PARAMETERS
        z_resolution = 1e3
        tolerance = 1e-8
        verbose_raman = 2

        # NLI PARAMETERS
        f_resolution_nli = 0.2e9
        verbose_nli = 1
        method_nli = 'GGN_spectrally_separated_spm_xpm'
        n_points_per_slot_min = 10
        n_points_per_slot_max = 5000
        delta_f = 50e9
        min_fwm_inv = 60
        dense_regime = namedtuple('DenseRegimeParameters',
                                  'n_points_per_slot_min n_points_per_slot_max delta_f min_fwm_inv')
        dense_regime = dense_regime(n_points_per_slot_min=n_points_per_slot_min,
                                    n_points_per_slot_max=n_points_per_slot_max, delta_f=delta_f,
                                    min_fwm_inv=min_fwm_inv)

        # FIBER
        fiber_info = namedtuple('FiberInformation', 'length attenuation_coefficient raman_coefficient '
                                                    'beta2 beta3 f_ref_beta gamma')
        attenuation_coefficient = namedtuple('AttenuationCoefficient', 'alpha_power frequency')
        raman_coefficient = namedtuple('RamanCoefficient', 'cr frequency')

        att_coeff = attenuation_coefficient(alpha_power=attenuation_coefficient_p, frequency=frequency_attenuation)
        raman_coeff = raman_coefficient(cr=cr, frequency=frequency_cr)
        fiber = fiber_info(length=fiber_length, attenuation_coefficient=att_coeff, raman_coefficient=raman_coeff,
                           gamma=gamma, beta2=beta2, beta3=beta3, f_ref_beta=f_ref_beta)

        # SPECTRUM
        spectral_information = namedtuple('SpectralInformation', 'carriers')
        channel = namedtuple('Channel', 'channel_number frequency baud_rate roll_off power')
        power = namedtuple('Power', 'signal nonlinear_interference amplified_spontaneous_emission')

        carriers = tuple(channel(1 + ii, start_f + (delta_f * ii), symbol_rate, roll_off, power(pch, 0, 0))
                         for ii in range(0, num_channels))
        spectrum = spectral_information(carriers=carriers)

        # SOLVER PARAMETERS
        raman_solver_information = namedtuple('RamanSolverInformation', ' z_resolution tolerance verbose')
        solver_parameters = raman_solver_information(z_resolution=z_resolution,
                                                     tolerance=tolerance, verbose=verbose_raman)

        # NLI PARAMETERS
        nli_parameters = namedtuple('NLIParameters', 'method frequency_resolution verbose dense_regime')
        model_params = nli_parameters(method=method_nli, frequency_resolution=f_resolution_nli,
                                      verbose=verbose_nli, dense_regime=dense_regime)

        raman_solver = rm.RamanSolver(fiber)
        raman_solver.spectral_information = spectrum
        raman_solver.solver_params = solver_parameters

        nlint = nli.NLI(fiber_information=fiber)
        nlint.srs_profile = raman_solver
        nlint.model_parameters = model_params

        carriers_nli_ggn = [nlint.compute_nli(carrier, *spectrum.carriers)
                        for carrier in spectrum.carriers
                        if carrier.channel_number in cut_list]

        method_nli = 'GN_spectrally_separated_spm_xpm'
        nli_parameters = namedtuple('NLIParameters', 'method frequency_resolution verbose dense_regime')
        model_params = nli_parameters(method=method_nli, frequency_resolution=f_resolution_nli,
                                      verbose=verbose_nli, dense_regime=dense_regime)
        nlint.model_parameters = model_params

        carriers_nli_gn = [nlint.compute_nli(carrier, *spectrum.carriers)
                        for carrier in spectrum.carriers
                        if carrier.channel_number in cut_list]
        npt.assert_allclose(carriers_nli_ggn, carriers_nli_gn, rtol=1E-12)
#
# def test_nli_spectral_separated():
#     # FIBER PARAMETERS
#     fiber_information = namedtuple('FiberInformation',
#                                    'length attenuation_coefficient raman_coefficient beta2 beta3 gamma')
#     attenuation_coefficient_p = namedtuple('Attenuation_coeff', 'alpha_power')
#     attenuation_coefficient_p.alpha_power = np.array([np.log(10) * 0.18895E-3 / 10])
#     fiber_information.attenuation_coefficient = attenuation_coefficient_p
#     fiber_information.beta2 = -21.27e-27  # s^2/m
#     fiber_information.beta3 = 0  # s^3/m
#     fiber_information.f_ref_beta = 0  # Hz
#     fiber_information.gamma = 1.3e-3  # 1/W/m
#
#     model_parameters = namedtuple('NLIParameters', 'method frequency_resolution verbose')
#     model_parameters.method = 'GGN_spectrally_separated_xpm_spm'
#     model_parameters.frequency_resolution = 0.1e9
#     model_parameters.verbose = False
#
#     # WDM COMB PARAMETERS
#     roll_off = 0.1
#     symbol_rate = 32e9
#
#     # SPECTRUM
#     spectral_information = namedtuple('SpectralInformation', 'carriers')
#     channel = namedtuple('Channel', 'channel_number frequency baud_rate roll_off power')
#     power = namedtuple('Power', 'signal nonlinear_interference amplified_spontaneous_emission')
#
#     csv_files_dir = './tests/resources/'
#     f_axis = np.loadtxt(open(csv_files_dir + 'f_axis.csv', 'rb'), delimiter=',') * 1E+12
#     z_array = np.loadtxt(open(csv_files_dir + 'z_array.csv', 'rb'), delimiter=',') * 1E+3
#     rho = np.loadtxt(open(csv_files_dir + 'raman_profile.csv'), delimiter=',')
#     A = np.exp((-attenuation_coefficient_p.alpha_power / 2) * z_array)
#     for i in range(len(rho)):
#         rho[i] = np.multiply(rho[i], A)
#     f_channel = np.loadtxt(open(csv_files_dir + 'f_channel.csv', 'rb'), delimiter=',') * 1E+12
#     # f_channel = f_channel[range(11)]
#     l = len(f_channel)
#     cut_number = [5, 23, 40, 57, 73, 84, 102, 120, 138, 156]
#     pch = 0.50119E-03 * np.ones(l)
#     channel_numbers = range(l)
#     carriers = tuple(channel(i + 1, f_channel[i], symbol_rate, roll_off, power(pch[i], 0, 0)) for i in channel_numbers)
#     spectrum = spectral_information(carriers=carriers)
#     raman_solver = namedtuple('RamanSolver', 'stimulated_raman_scattering spectral_information')
#     raman_solver.spectral_information = spectrum
#     stimulated_raman_scattering = namedtuple('stimulated_raman_scattering', ' rho z frequency ')
#     stimulated_raman_scattering = stimulated_raman_scattering(rho=rho, z=z_array, frequency=f_axis)
#     raman_solver = raman_solver(stimulated_raman_scattering=stimulated_raman_scattering, spectral_information=spectrum)
#
#     nlint = nli.NLI(fiber_information=fiber_information)
#     nlint.srs_profile = raman_solver
#     nlint.model_parameters = model_parameters
#
#     # Compute RAMAN SRS
#     rho_end = interp1d(raman_solver.stimulated_raman_scattering.frequency,
#                        raman_solver.stimulated_raman_scattering.rho[:, -1])
#
#     # OUTPUT VS EXPECTED
#     expected_snr_nl = [35.642634, 35.212572, 35.010348, 34.879301, 34.785609, 34.733054,
#                        34.704074, 34.710039, 34.758298, 34.891119, 35.27643]
#     counter = 0
#     for carrier in carriers:
#         if carrier.channel_number in cut_number:
#             carrier_nli = nlint.compute_nli(carrier, *carriers)
#             p_cut = carrier.power.signal
#             f_cut = carrier.frequency
#             p_cut = np.array(p_cut)  # * (rho_end(f_cut)) ** 2
#             snr_nl = 10 * np.log10(p_cut / carrier_nli)
#             npt.assert_allclose(snr_nl, expected_snr_nl[counter], rtol=1E-6)
#             counter += 1
