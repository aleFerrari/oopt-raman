
import csv
import numpy as np
from collections import namedtuple
from raman import nli
import raman.raman as rm
import matplotlib.pyplot as plt
from operator import attrgetter
from scipy.interpolate import interp1d

cut_index = range(1,11);

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


def main(fiber_information, spectral_information, raman_solver, model_params):
    nlint = nli.NLI(fiber_information=fiber_information)
    nlint.srs_profile = raman_solver
    nlint.model_parameters = model_params


    carriers_nli = [nlint.compute_nli(carrier, *spectral_information.carriers)
                    for carrier in spectral_information.carriers if (carrier.channel_number in cut_index)]

    return carriers_nli


if __name__ == '__main__':

    # FIBER PARAMETERS
    cr_file_name = './raman_gain_efficiency/SSMF.csv'
    cr, frequency_cr = raman_gain_efficiency_from_csv(cr_file_name)

    fiber_length = np.array([80e3])
    attenuation_coefficient_p = np.array([np.log(10)*0.18895E-3/10])
    frequency_attenuation = np.array([193.5e12])

    gamma = 1.3e-3     # 1/W/m
    beta2 = -21.27e-27   # s^2/m
    beta3 = 0  # s^3/m

    # WDM COMB PARAMETERS
    num_channels = 91
    delta_f = 50e9
    pch = 1e-3
    roll_off = 0.1
    symbol_rate = 32e9
    start_f = 191.0e12

    # RAMAN PUMP PARAMETERS
    pump_pow = [150e-3, 250e-3, 150e-3, 250e-3, 200e-3]
    pump_freq = [200.2670e12, 201.6129e12, 207.1823e12, 208.6231e12, 210.0840e12]
    prop_direction = [-1, -1, -1, -1, -1]
    num_pumps = len(pump_pow)

    # ODE SOLVER PARAMETERS
    z_resolution = 1e3
    tolerance = 1e-8
    verbose_raman = 2

    # NLI PARAMETERS
    f_resolution_nli = 1e9
    verbose_nli = 1
    method_nli = 'ggn_integral'

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

    # RAMAN PUMPS
    raman_pump_information = namedtuple('SpectralInformation', 'raman_pumps')
    pump = namedtuple('RamanPump', 'pump_number power frequency propagation_direction')
    pumps = tuple(pump(1 + ii, pump_pow[ii], pump_freq[ii], prop_direction[ii])
                  for ii in range(0, num_pumps))
    raman_pumps = raman_pump_information(raman_pumps=pumps)

    # SOLVER PARAMETERS
    raman_solver_information = namedtuple('RamanSolverInformation', ' z_resolution tolerance verbose')
    solver_parameters = raman_solver_information(z_resolution=z_resolution,
                                                 tolerance=tolerance, verbose=verbose_raman)

    # NLI PARAMETERS
    nli_parameters = namedtuple('NLIParameters', 'method frequency_resolution verbose')
    model_params = nli_parameters(method=method_nli, frequency_resolution=f_resolution_nli, verbose=verbose_nli)

    raman_solver = rm.RamanSolver(fiber)
    raman_solver.spectral_information = spectrum
    raman_solver.raman_pump_information = raman_pumps
    raman_solver.solver_params = solver_parameters

    # COMPARISON WITH MATHLAB RESULTS

    csv_files_dir = './resources/'

    f_axis = np.array([187.7000, 187.7500, 187.8000, 187.8500, 187.9000, 187.9500, 188.0000, 188.0500, 188.1000, 188.1500])
    z_array = (1E+3)*np.loadtxt(open(csv_files_dir+'z_array.csv','rb'),delimiter=',')

    z_array = (1E+3) * np.array(range(81))
    rho = np.loadtxt(open(csv_files_dir+'raman_profile.csv'),delimiter=',')
    A = np.exp((-attenuation_coefficient_p / 2) * z_array)
    for i in range(len(rho)):
        rho[i] = np.multiply(rho[i], A)

    guard_band_indices = range(78, 83)
    f_channel = np.array([187.7000, 187.7500, 187.8000, 187.8500, 187.9000, 187.9500, 188.0000, 188.0500, 188.1000, 188.1500])
    pch = 0.50119E-03*np.ones(len(f_channel))
    channel_numbers = range(len(f_channel))


    carriers = tuple(channel(i+1,f_channel[i],symbol_rate,roll_off,power(pch[i],0,0)) for i in channel_numbers)
    spectrum = spectral_information(carriers=carriers)
    raman_solver.spectral_information = spectrum
    raman_solver = namedtuple('RamanSolver','raman_bvp_solution spectral_information')
    raman_bvp_solution = namedtuple('raman_bvp_solution',' rho z frequency ')
    raman_bvp_solution = raman_bvp_solution(rho=rho,z=z_array,frequency=f_axis)
    raman_solver = raman_solver(raman_bvp_solution=raman_bvp_solution,spectral_information=spectrum)

    carriers_nli = main(fiber, spectrum, raman_solver, model_params)

    # PLOT RESULTS
    p_cut = [carrier.power.signal for carrier in sorted(spectrum.carriers, key=attrgetter('frequency')) if (carrier.channel_number in cut_index)]
    f_cut = [carrier.frequency for carrier in sorted(spectrum.carriers, key=attrgetter('frequency')) if (carrier.channel_number in cut_index)]

    rho_end = interp1d(raman_solver.raman_bvp_solution.frequency, raman_solver.raman_bvp_solution.rho[:,-1])
    p_cut = np.array(p_cut) * (rho_end(f_cut))**2

    snr_nl = p_cut / carriers_nli

    np.save('snr_2',snr_nl)
    np.save('carriers_nli_2',np.array(carriers_nli))

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
