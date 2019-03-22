import os
import datetime
import csv
import numpy as np
from collections import namedtuple
from raman import nli
import raman.utilities as ut
import raman.raman as rm
import matplotlib.pyplot as plt
from operator import attrgetter
from scipy.interpolate import interp1d


def raman_gain_efficiency_from_csv(csv_file_name):
    with open(csv_file_name) as csv_file:
        cr_data = csv.reader(csv_file, delimiter=',')
        next(cr_data, None)
        cr = np.array([])
        frequency_cr = np.array([])
        for row in cr_data:
            frequency_cr = np.append(frequency_cr, float(row[0]))
            cr = np.append(cr, float(row[1]))

    return cr, frequency_cr


def main(fiber_information, spectral_information, raman_solver, model_params, cut_list):
    raman_solver.stimulated_raman_scattering  # Compute SRS profile
    
    nlint = nli.NLI(fiber_information=fiber_information)
    nlint.srs_profile = raman_solver
    nlint.model_parameters = model_params

    carriers_nli = [nlint.compute_nli(carrier, *spectral_information.carriers)
                    for carrier in spectral_information.carriers
                    if carrier.channel_number in cut_list]

    return carriers_nli


if __name__ == '__main__':

    # FIBER PARAMETERS
    cr_file_name = './raman_gain_efficiency/SSMF.csv'
    cr, frequency_cr = raman_gain_efficiency_from_csv(cr_file_name)

    fiber_length = np.array([75e3])
    attenuation_coefficient_p = np.array([0.046e-3])
    frequency_attenuation = np.array([193.5e12])

    gamma = 1.27e-3     # 1/W/m
    beta2 = 21.27e-27   # s^2/m
    beta3 = 0.0344e-39   # s^3/m

    # WDM COMB PARAMETERS
    num_channels = 91
    delta_f = 50e9
    pch = 0.5e-3

    cut_list = range(5, num_channels, 5)
  
    roll_off = 0.1
    symbol_rate = 32e9
    start_f = 191.0e12

    # ODE SOLVER PARAMETERS
    z_resolution = 1e3
    tolerance = 1e-8
    verbose_raman = 2

    # NLI PARAMETERS
    f_resolution_nli = 2e9
    verbose_nli = 1
    method_nli = 'ggn_integral'
    n_points_per_slot_min = 4
    n_points_per_slot_max = 5000 
    delta_f = 50e9
    min_fwm_inv = 60
    dense_regime = namedtuple('DenseRegimeParameters', 'n_points_per_slot_min n_points_per_slot_max delta_f min_fwm_inv')
    dense_regime = dense_regime(n_points_per_slot_min=n_points_per_slot_min, n_points_per_slot_max=n_points_per_slot_max, delta_f=delta_f, min_fwm_inv=min_fwm_inv)

    # FIBER
    fiber_info = namedtuple('FiberInformation', 'length attenuation_coefficient raman_coefficient beta2 beta3 gamma')
    attenuation_coefficient = namedtuple('AttenuationCoefficient', 'alpha_power frequency')
    raman_coefficient = namedtuple('RamanCoefficient', 'cr frequency')

    att_coeff = attenuation_coefficient(alpha_power=attenuation_coefficient_p, frequency=frequency_attenuation)
    raman_coeff = raman_coefficient(cr=cr, frequency=frequency_cr)
    fiber = fiber_info(length=fiber_length, attenuation_coefficient=att_coeff, raman_coefficient=raman_coeff,
                                        gamma=gamma, beta2=beta2, beta3=beta3)

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
    model_params = nli_parameters(method=method_nli, frequency_resolution=f_resolution_nli, verbose=verbose_nli, dense_regime=dense_regime)

    raman_solver = rm.RamanSolver(fiber)
    raman_solver.spectral_information = spectrum
    raman_solver.solver_params = solver_parameters

    carriers_nli = main(fiber, spectrum, raman_solver, model_params, cut_list)

    # PLOT RESULTS
    p_cut = [carrier.power.signal for carrier in sorted(spectrum.carriers, key=attrgetter('frequency'))
             if carrier.channel_number in cut_list]
    f_cut = [carrier.frequency for carrier in sorted(spectrum.carriers, key=attrgetter('frequency'))
             if carrier.channel_number in cut_list]

    rho_end = interp1d(raman_solver.stimulated_raman_scattering.frequency, raman_solver.stimulated_raman_scattering.rho[:,-1])
    p_cut = np.array(p_cut) * (rho_end(f_cut))**2

    snr_nl = p_cut / carriers_nli

    fig1 = plt.figure()
    plt.plot(f_cut, 10*np.log10(p_cut)+30, '*')
    plt.plot(f_cut, 10*np.log10(carriers_nli)+30, '*')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power [dBm]')
    plt.grid()

    fig2 = plt.figure()
    plt.plot(f_cut, 10*np.log10(snr_nl), '-o')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'$SNR_{NL}$ [dB]')
    plt.grid()

    plt.show()
