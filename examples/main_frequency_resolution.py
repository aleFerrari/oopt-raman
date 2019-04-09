import os
import time
from collections import namedtuple
from raman import nli
import numpy as np
import scipy.io as sio
from numpy import pi


def get_fib_params():
    fiber_information = namedtuple('FiberInformation',
                                   'length attenuation_coefficient raman_coefficient beta2 beta3 gamma')
    attenuation_coefficient_p = namedtuple('Attenuation_coeff', 'alpha_power')
    attenuation_coefficient_p.alpha_power = np.array([np.log(10) * 0.18895E-3 / 10])
    fiber_information.attenuation_coefficient = attenuation_coefficient_p
    fiber_information.beta2 = -21.27e-27  # s^2/m
    fiber_information.beta3 = 0  # s^3/m
    fiber_information.f_ref_beta = 0  # Hz
    fiber_information.gamma = 1.3e-3  # 1/W/m

    return fiber_information


def get_spec_info(pch, f_channel, ch_number):
    l = len(f_channel)

    # WDM COMB PARAMETERS
    roll_off = 0
    symbol_rate = 32e9

    spectral_information = namedtuple('SpectralInformation', 'carriers')
    channel = namedtuple('Channel', 'channel_number frequency baud_rate roll_off power')
    power = namedtuple('Power', 'signal nonlinear_interference amplified_spontaneous_emission')
    channel_numbers = range(l)
    carriers = tuple(
        channel(i + 1, f_channel[i], symbol_rate, roll_off, power(pch[i], 0, 0))
        for i in channel_numbers
        if (i+1 in ch_number))
    spectrum = spectral_information(carriers=carriers)

    return spectrum


def frequency_resolution(carrier, carriers, grid_size, alpha_ef, delta_z, beta2, k_tol, phi_tol):

    f_pump_resolution, method_f_pump, res_dict_pump = \
        _get_freq_res_k_phi(0, grid_size, alpha_ef, delta_z, beta2, k_tol, phi_tol)

    f_cut_resolution = {}
    method_f_cut = {}
    res_dict_cut = {}
    for cut_carrier in carriers:
        delta_number = cut_carrier.channel_number - carrier.channel_number
        delta_count = abs(delta_number)
        f_res, method, res_dict = \
            _get_freq_res_k_phi(delta_count, grid_size, alpha_ef, delta_z, beta2, k_tol, phi_tol)
        f_cut_resolution[f'delta_{delta_number}'] = f_res
        method_f_cut[delta_number] = method
        res_dict_cut[delta_number] = res_dict

    return [f_cut_resolution, f_pump_resolution, (method_f_cut, method_f_pump), (res_dict_cut, res_dict_pump)]


def _get_freq_res_k_phi(delta_count, grid_size, alpha_ef, delta_z, beta2, k_tol, phi_tol):

    res_phi = _get_freq_res_phase_rotation(delta_count, grid_size, delta_z, beta2, phi_tol)
    res_k = _get_freq_res_dispersion_attenuation(delta_count, grid_size, alpha_ef, beta2, k_tol)

    res_dict = {'res_phi': res_phi, 'res_k': res_k}
    method = min(res_dict, key=res_dict.get)

    return res_dict[method], method, res_dict


def _get_freq_res_dispersion_attenuation(delta_count, grid_size, alpha_ef, beta2, k_tol):
    return k_tol * 2 * abs(alpha_ef) / abs(beta2) / (1 + delta_count) / (4*pi**2 * grid_size)


def _get_freq_res_phase_rotation(delta_count, grid_size, delta_z, beta2, phi_tol):
    return phi_tol / abs(beta2) / (1 + delta_count) / delta_z / (4*pi**2 * grid_size)


if __name__=="__main__":
    csv_files_dir = './resources/'
    folder_results = './results/frequency_resolution/'
    flag_save = True
    flag_raman = False

    flag_single_resolution = False
    single_res = 2e9

    cut_number = [120]
    pump_number = [70, 80, 90, 100, 110, 115, 116, 117, 118, 119, 120]
    #pump_number = [70, 110, 118, 119, 120]
    pump_number = list(range(0, 170))

    # cut_number = [40]
    # pump_number = [40, 41, 42, 43, 44, 45, 50, 60, 70, 80, 100, 120, 140, 160]
    # pump_number = [40, 41]

    phi_list = [0.1, 0.01, 0.04, 0.001]
    k_list = [3, 2, 1.5, 1, 0.5, 0.1]
    phi_list = [0.1]
    k_list = [1]
    phi_list = [0.04]
    #k_list = [1.5 * 1e3]
    #phi_list = [1 * 1e30]

    f_channel = np.loadtxt(csv_files_dir + 'f_channel.csv', delimiter=',') * 1E+12
    l = len(f_channel)
    pch = 0.50119E-03 * np.ones(l)

    fiber_information = get_fib_params()
    channel_numbers = sorted(pump_number+cut_number)
    spectrum = get_spec_info(pch, f_channel, channel_numbers)
    carriers = spectrum.carriers

    f_axis = np.loadtxt(csv_files_dir + 'f_axis.csv', delimiter=',') * 1E+12
    z_array = np.loadtxt(csv_files_dir + 'z_array.csv', delimiter=',') * 1E+3
    fiber_information.length = z_array[-1]
    rho = np.loadtxt(csv_files_dir + 'raman_profile.csv', delimiter=',')
    if not flag_raman:
        rho = rho * 0 + 1
    A_ef = np.exp((-fiber_information.attenuation_coefficient.alpha_power / 2) * z_array)
    for i in range(len(rho)):
        rho[i] = np.multiply(rho[i], A_ef)
    raman_solver = namedtuple('RamanSolver', 'stimulated_raman_scattering spectral_information')
    raman_solver.spectral_information = spectrum
    stimulated_raman_scattering = namedtuple('stimulated_raman_scattering', ' rho z frequency ')
    stimulated_raman_scattering = stimulated_raman_scattering(rho=rho, z=z_array, frequency=f_axis)
    raman_solver = raman_solver(stimulated_raman_scattering=stimulated_raman_scattering,
                                spectral_information=spectrum)

    grid_size = 50e9
    alpha_ef = fiber_information.attenuation_coefficient.alpha_power / 2
    beta2 = fiber_information.beta2
    delta_z = z_array[1] - z_array[0]
    for phi in phi_list:
        for k_tol in k_list:

            model_parameters = namedtuple('NLIParameters',
                                          'method frequency_resolution verbose dense_regime spectral_sep')
            model_parameters.method = 'GGN_spectrally_separated_xpm'
            model_parameters.frequency_resolution = 1e12
            model_parameters.verbose = True

            n_points_per_slot_min = 4
            n_points_per_slot_max = 1000
            delta_f = 50e9
            min_fwm_inv = 60
            dense_regime = namedtuple('DenseRegimeParameters',
                                      'n_points_per_slot_min n_points_per_slot_max delta_f min_fwm_inv')
            dense_regime = dense_regime(n_points_per_slot_min=n_points_per_slot_min,
                                        n_points_per_slot_max=n_points_per_slot_max, delta_f=delta_f,
                                        min_fwm_inv=min_fwm_inv)
            model_parameters.dense_regime = dense_regime

            counter = 0
            eta_list = []
            p_cut = []
            start = time.time()
            for carrier in carriers:
                if carrier.channel_number in cut_number:

                    resolution_param = \
                        frequency_resolution(carrier, carriers, grid_size, alpha_ef, delta_z, beta2, k_tol, phi)
                    f_cut_resolution, f_pump_resolution, (method_f_cut, method_f_pump), (res_dict_cut, res_dict_pump) =\
                        resolution_param

                    spectral_sep = namedtuple('SpectralSep', 'f_cut_resolution f_pump_resolution')

                    if flag_single_resolution:
                        f_pump_resolution = single_res
                        for key in f_cut_resolution:
                            f_cut_resolution[key] = single_res
                        resolution_param[0] = f_cut_resolution
                        resolution_param[1] = f_pump_resolution

                    spectral_sep.f_cut_resolution = f_cut_resolution
                    spectral_sep.f_pump_resolution = f_pump_resolution
                    model_parameters.spectral_sep = spectral_sep

                    nlint = nli.NLI(fiber_information=fiber_information)
                    nlint.srs_profile = raman_solver
                    nlint.model_parameters = model_parameters

                    eta_list.append(nlint._compute_eta_matrix(carrier, *carriers))
                    p_cut.append(carrier.power.signal)
                    f_cut = carrier.frequency
                    counter += 1
            p_cut = np.array(p_cut)
            comp_time = time.time() - start
            print(f'Computed in {comp_time} seconds')

            # SAVE DATA
            if not os.path.exists(folder_results):
                os.makedirs(folder_results)

            # SAVE
            if flag_save:
                data = {'eta': eta_list, 'cut_index': cut_number, 'p_ch': pch, 'time': comp_time, 'phi': phi,
                        'k_tol': k_tol, 'resolution_parameters': resolution_param}
                sio.savemat(folder_results +
                            f'timestamp{time.time()}_{model_parameters.method}cut={cut_number}_eta_matix.mat',
                            data)

            print('')