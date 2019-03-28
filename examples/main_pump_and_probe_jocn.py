import os
import numpy as np
import csv
from numpy import testing as npt
import pytest
from raman import nli
from raman import raman as rm
from collections import namedtuple
from scipy.interpolate import interp1d
import time

folder_results = './results/pump_and_probe_jocn/'

# FIBER PARAMETERS
fiber_information = namedtuple('FiberInformation',
                               'length attenuation_coefficient raman_coefficient beta2 beta3 gamma')
attenuation_coefficient_p = namedtuple('Attenuation_coeff', 'alpha_power')
attenuation_coefficient_p.alpha_power = np.array([np.log(10) * 0.18895E-3 / 10])
fiber_information.attenuation_coefficient = attenuation_coefficient_p
fiber_information.beta2 = -21.27e-27  # s^2/m
fiber_information.beta3 = 0  # s^3/m
fiber_information.f_ref_beta = 0  # Hz
fiber_information.gamma = 1.3e-3  # 1/W/m

model_parameters = namedtuple('NLIParameters', 'method frequency_resolution verbose dense_regime')
model_parameters.method = 'GN_spectrally_separated_spm_xpm'
model_parameters.frequency_resolution = 0.2e9
model_parameters.verbose = True
n_points_per_slot_min = 4
n_points_per_slot_max = 1000
delta_f = 50e9
min_fwm_inv = 60
dense_regime = namedtuple('DenseRegimeParameters',
                          'n_points_per_slot_min n_points_per_slot_max delta_f min_fwm_inv')
dense_regime = dense_regime(n_points_per_slot_min=n_points_per_slot_min,
                            n_points_per_slot_max=n_points_per_slot_max, delta_f=delta_f, min_fwm_inv=min_fwm_inv)
model_parameters.dense_regime = dense_regime

# WDM COMB PARAMETERS
roll_off = 0.1
symbol_rate = 32e9

# SPECTRUM
spectral_information = namedtuple('SpectralInformation', 'carriers')
channel = namedtuple('Channel', 'channel_number frequency baud_rate roll_off power')
power = namedtuple('Power', 'signal nonlinear_interference amplified_spontaneous_emission')

csv_files_dir = './resources/'
f_axis = np.loadtxt(open(csv_files_dir + 'f_axis.csv', 'rb'), delimiter=',') * 1E+12
z_array = np.loadtxt(open(csv_files_dir + 'z_array.csv', 'rb'), delimiter=',') * 1E+3
fiber_information.length = z_array[-1]
rho = np.loadtxt(open(csv_files_dir + 'raman_profile.csv'), delimiter=',') #*0 +1
A_ef = np.exp((-attenuation_coefficient_p.alpha_power / 2) * z_array)
for i in range(len(rho)):
    rho[i] = np.multiply(rho[i], A_ef)
f_channel = np.loadtxt(open(csv_files_dir + 'f_channel.csv', 'rb'), delimiter=',') * 1E+12
l = len(f_channel)
cut_number = [5, 23, 40, 57, 73, 84, 102, 120, 138, 156]
pch = 0.50119E-03 * np.ones(l)
channel_numbers = range(l)
carriers = tuple(channel(i + 1, f_channel[i], symbol_rate, roll_off, power(pch[i], 0, 0)) for i in channel_numbers)
spectrum = spectral_information(carriers=carriers)
raman_solver = namedtuple('RamanSolver', 'stimulated_raman_scattering spectral_information')
raman_solver.spectral_information = spectrum
stimulated_raman_scattering = namedtuple('stimulated_raman_scattering', ' rho z frequency ')
stimulated_raman_scattering = stimulated_raman_scattering(rho=rho, z=z_array, frequency=f_axis)
raman_solver = raman_solver(stimulated_raman_scattering=stimulated_raman_scattering, spectral_information=spectrum)

nlint = nli.NLI(fiber_information=fiber_information)
nlint.srs_profile = raman_solver
nlint.model_parameters = model_parameters

# OUTPUT VS EXPECTED
snr_nl_split_step = [25.3415, 23.7746, 22.9571, 23.0103, 23.7999, 24.5674, 25.3280, 25.8465, 26.2307, 26.3494]
expected_snr_nl = snr_nl_split_step
# counter = 0
# for carrier in carriers:
#     if carrier.channel_number in cut_number:
#         carrier_nli = nlint.compute_nli(carrier, *carriers)
#         p_cut = carrier.power.signal
#         f_cut = carrier.frequency
#         p_cut = np.array(p_cut)  # * (rho_end(f_cut)) ** 2
#         snr_nl = 10 * np.log10(p_cut / carrier_nli) - 10*np.log10(10)  # 10 spans
#         npt.assert_allclose(snr_nl, expected_snr_nl[counter], rtol=1E-6)
#         counter += 1
counter = 0
carrier_nli = []
p_cut = []
start = time.time()
for carrier in carriers:
    if carrier.channel_number in cut_number:
        carrier_nli.append(nlint.compute_nli(carrier, *carriers))
        p_cut.append(carrier.power.signal)
        f_cut = carrier.frequency
        counter += 1
p_cut = np.array(p_cut)
carrier_nli = np.array(carrier_nli)
comp_time = time.time() - start
snr_nl = 10 * np.log10(p_cut / carrier_nli) - 10*np.log10(10)  # 10 span
print(f'Computed in {comp_time} seconds')
print(snr_nl)

# SAVE DATA
if not os.path.exists(folder_results):
    os.makedirs(folder_results)

# SNR from np.array to list
snr_nl = [elem for elem in snr_nl]

with open(folder_results + 'snr_nl.csv', 'a') as fd:
    writer = csv.writer(fd)
    writer.writerow(['Method'] + [model_parameters.method])
    writer.writerow(['f resolution [Hz]'] + [model_parameters.frequency_resolution])
    writer.writerow(['n_points_per_slot_max'] + [model_parameters.dense_regime.n_points_per_slot_max])
    writer.writerow(['n_points_per_slot_min'] + [model_parameters.dense_regime.n_points_per_slot_min])
    writer.writerow(['delta_f [Hz]'] + [model_parameters.dense_regime.delta_f])
    writer.writerow(['min_fwm_inv'] + [model_parameters.dense_regime.min_fwm_inv])
    writer.writerow(['SNR NL [dB]'] + snr_nl)
    writer.writerow([])
