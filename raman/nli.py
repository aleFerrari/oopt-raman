# -*- coding: utf-8 -*-

"""
===============
This module contains the class NLI to compute the nonlinear interference introduced by the fiber.
@Author: Alessio Ferrari
"""
import progressbar
import numpy as np
import raman.utilities as ut
from operator import attrgetter
from scipy.interpolate import interp1d

class NLI:
    """ This class implements the NLI models.
        Model and method can be specified in `self.model_parameters.method`.
        List of implemented methods:
        'GGN_integral': brute force triple integral solution
        'GGN_spectrally_separated_xpm_spm': XPM plus SPM
    """

    def __init__(self, fiber_information=None):
        """ Initialize the fiber object with its physical parameters
        """
        self.fiber_information = fiber_information
        self.srs_profile = None
        self.model_parameters = None

    @property
    def fiber_information(self):
        return self._fiber_information

    @fiber_information.setter
    def fiber_information(self, fiber_information):
        self._fiber_information = fiber_information

    @property
    def srs_profile(self):
        return self._srs_profile

    @srs_profile.setter
    def srs_profile(self, srs_profile):
        self._srs_profile = srs_profile

    @property
    def model_parameters(self):
        return self._model_parameters

    @model_parameters.setter
    def model_parameters(self, model_params):
        """
        :param model_params: namedtuple containing the parameters used to compute the NLI.
        """
        self._model_parameters = model_params

    def alpha0(self, f_eval=193.5e12):
        if len(self.fiber_information.attenuation_coefficient.alpha_power) == 1:
            alpha0 = self.fiber_information.attenuation_coefficient.alpha_power[0]
        else:
            alpha_interp = interp1d(self.fiber_information.attenuation_coefficient.frequency,
                                    self.fiber_information.attenuation_coefficient.alpha_power)
            alpha0 = alpha_interp(f_eval)
        return alpha0

    def _compute_dense_regimes(self, f1, f_eval, frequency_psd, len_carriers):
        f_central = min(frequency_psd) + (max(frequency_psd) - min(frequency_psd)) / 2
        frequency_psd = frequency_psd - f_central
        f_eval = f_eval - f_central
        f1 = f1 - f_central

        dense_regime = self.model_parameters.dense_regime
        n_points_per_slot_min = dense_regime.n_points_per_slot_min
        n_points_per_slot_max = dense_regime.n_points_per_slot_max
        delta_f = dense_regime.delta_f
        min_fwm_inv = 10 ** (dense_regime.min_fwm_inv/10)
        delta_f_min = delta_f / n_points_per_slot_min
        delta_f_max = delta_f / n_points_per_slot_max
        b_opt = max([len_carriers * delta_f, max(frequency_psd) - min(frequency_psd) + delta_f])
        f_max = 0.6 * b_opt
        alpha_e = (self.fiber_information.attenuation_coefficient.alpha_power / 2)
        beta2 = self.fiber_information.beta2

        if f1 == f_eval:
            f2dense_low_limit = -f_max
            f2dense_up_limit = f_max
        else:
            f2dense_up_limit = max(
                [f_eval + np.sqrt(alpha_e ** 2 / (4 * (np.pi ** 4) * (beta2 ** 2)) * (min_fwm_inv - 1)) / (f1 - f_eval),
                 f_eval - np.sqrt(alpha_e ** 2 / (4 * (np.pi ** 4) * (beta2 ** 2)) * (min_fwm_inv - 1)) / (f1 - f_eval)])
            # Limit on f2 based on classic FWM
            f2dense_low_limit = min(
                [f_eval + np.sqrt(alpha_e ** 2 / (4 * (np.pi ** 4) * (beta2 ** 2)) * (min_fwm_inv - 1)) / (f1 - f_eval),
                 f_eval - np.sqrt(alpha_e ** 2 / (4 * (np.pi ** 4) * (beta2 ** 2)) * (min_fwm_inv - 1)) / (f1 - f_eval)])

        if f2dense_low_limit == 0:
            f2dense_low_limit = -delta_f_min

        if f2dense_up_limit == 0:
            f2dense_up_limit = delta_f_min

        if f2dense_low_limit < -f_max:
            f2dense_low_limit = -f_max

        if f2dense_up_limit > f_max:
            f2dense_up_limit = f_max

        f2dense_width = abs(f2dense_up_limit - f2dense_low_limit)
        n_points_dense = np.ceil(
            f2dense_width / delta_f_min)  # Number of integration points to be considered in the denser area
        if n_points_dense < 100:
            n_points_dense = 100

        delta_f_array = f2dense_width / n_points_dense  # Get frequency spacing
        f2_array_dense = np.arange(f2dense_low_limit, f2dense_up_limit, delta_f_array)  # Define the denser grid

        if f_eval < 0:
            k = b_opt / 2 / (b_opt / 2 - delta_f_max)  # Get step ratio for logspace array definition
            n_log_short = np.ceil(np.log(f_max / abs(f2dense_low_limit)) * 1 / np.log(
                k) + 1)  # Get number of points required to ensure that the maximum frequency step in b_opt is not passed
            f2short = -(abs(f2dense_low_limit) * k ** (np.arange(n_log_short, 0, -1) - 1))  # Generate logspace array
            k = (b_opt / 2 + (abs(f2dense_up_limit) - f2dense_up_limit)) / (
                    b_opt / 2 - delta_f_max + (abs(f2dense_up_limit) - f2dense_up_limit))
            n_log_long = np.ceil(
                np.log((f_max + (abs(f2dense_up_limit) - f2dense_up_limit)) / abs(f2dense_up_limit)) * 1 / np.log(
                    k) + 1)  # Get number of points required to ensure that the maximum frequency step in b_opt is not passed
            f2long = (abs(f2dense_up_limit) * k ** (np.arange(1, n_log_long + 1, 1) - 1)) - (
                    abs(f2dense_up_limit) - f2dense_up_limit)
            f2array = np.array(list(f2short) + list(f2_array_dense[1:]) + list(f2long))
        else:
            k = b_opt / 2 / (b_opt / 2 - delta_f_max)
            n_log_short = np.ceil(np.log(f_max / abs(f2dense_up_limit)) * 1 / np.log(k) + 1)
            f2short = f2dense_up_limit * k ** (np.arange(1, n_log_short + 1, 1) - 1)
            k = (b_opt / 2 + (abs(f2dense_low_limit) + f2dense_low_limit)) / (
                    b_opt / 2 - delta_f_max + (abs(f2dense_low_limit) + f2dense_low_limit))
            n_log_long = np.ceil(
                np.log((f_max + (abs(f2dense_low_limit) + f2dense_low_limit)) / abs(f2dense_low_limit)) * 1 / np.log(
                    k) + 1)
            f2long = -(abs(f2dense_low_limit) * k ** (np.arange(n_log_long, 0, -1) - 1)) + (
                    abs(f2dense_low_limit) + f2dense_low_limit)
            f2array = np.array(list(f2long) + list(f2_array_dense[1:]) + list(f2short))

        return f2array + f_central

    def _verify_srs_wdm_comb(self, *carriers):
        """ Verify if SRS profile is associated to SRS
        """
        if len(carriers) != len(self.srs_profile.spectral_information.carriers):
            raise ValueError(f'Number of carriers of `self.srs_profile` is '
                             f'{len(self.srs_profile.spectral_information.carriers)},'
                             f'while number of carriers in `carriers` is {len(carriers)}.')

        for index, srs_carrier in enumerate(self.srs_profile.spectral_information.carriers):
            if (srs_carrier.power.signal != carriers[index].power.signal) or \
               (srs_carrier.frequency != carriers[index].frequency):
                raise ValueError(f'Carrier #{carriers[index].channel_number} of self.srs_profile does not match '
                                 f'with #{srs_carrier.channel_number} carrier in *carriers')

    def compute_nli(self, carrier, *carriers):
        """
        Compute NLI power generated by the WDM comb `*carriers` on the channel under test `carrier` 
        at the end of the fiber span.
        """
        if self.srs_profile:
            self._verify_srs_wdm_comb(*carriers)

        if 'ggn_integral' == self.model_parameters.method.lower():
            carrier_nli = self._compute_ggn_integral(carrier, *carriers)
        elif 'gn_analytic' == self.model_parameters.method.lower():
            carrier_nli = self._gn_analytic(carrier, *carriers)
        elif 'spectrally_separated' in self.model_parameters.method.lower():
            eta_matrix = self._compute_eta_matrix(carrier, *carriers)
            carrier_nli = self._carrier_nli_from_eta_matrix(eta_matrix, carrier, *carriers)
        else:
            raise ValueError(f'Method {self.model_parameters.method_nli} not implemented.')

        return carrier_nli

    @staticmethod
    def _carrier_nli_from_eta_matrix(eta_matrix, carrier, *carriers):
        carrier_nli = 0
        for pump_carrier_1, eta_row in zip(carriers, eta_matrix):
            for pump_carrier_2, eta in zip(carriers, eta_row):
                carrier_nli += eta * pump_carrier_1.power.signal * pump_carrier_2.power.signal
        carrier_nli *= carrier.power.signal

        return carrier_nli

    def _compute_eta_matrix(self, carrier_cut, *carriers):

        cut_index = carrier_cut.channel_number - 1
        # Matrix initialization
        eta_matrix = np.zeros(shape=(len(carriers), len(carriers)))

        # SPM
        if 'spm' in self.model_parameters.method.lower():
            if self.model_parameters.verbose:
                print(f'Start computing SPM on channel #{carrier_cut.channel_number}')
            # SPM GGN
            if 'ggn' in self.model_parameters.method.lower():
                partial_nli = self._generalized_spectrally_separated_spm(carrier_cut)
            # SPM GN
            elif 'gn' in self.model_parameters.method.lower():
                partial_nli = self._gn_analytic(carrier_cut, *[carrier_cut])
            eta_matrix[cut_index, cut_index] = partial_nli / carrier_cut.power.signal**3

        # XPM
        if 'xpm' in self.model_parameters.method.lower():
            for pump_index, pump_carrier in enumerate(carriers):
                if not (cut_index == pump_index):
                    if self.model_parameters.verbose:
                        print(f'Start computing XPM on channel #{carrier_cut.channel_number} '
                              f'from channel #{pump_carrier.channel_number}')
                    # spectrally separated GGN
                    if 'ggn' in self.model_parameters.method.lower():
                        partial_nli = self._generalized_spectrally_separated_xpm(carrier_cut, pump_carrier)
                    elif 'gn' in self.model_parameters.method.lower():
                        partial_nli = self._gn_analytic(carrier_cut, *[pump_carrier]) /\
                                                             (carrier_cut.power.signal * carrier_cut.power.signal**2)
                    eta_matrix[pump_index, pump_index] = partial_nli
        return eta_matrix

    # Methods for computing spectrally separated GGN
    def _generalized_spectrally_separated_spm(self, carrier):
        eta = (16.0 / 27.0) * self.fiber_information.gamma**2 * carrier.baud_rate *\
              2 * self._generalized_psi(carrier, carrier)

        return eta

    def _generalized_spectrally_separated_xpm(self, carrier_cut, pump_carrier):
        eta = (16.0 / 27.0) * self.fiber_information.gamma**2 * carrier_cut.baud_rate *\
              self._generalized_psi(carrier_cut, pump_carrier)

        return eta

    def _generalized_psi(self, carrier_cut, pump_carrier):
        """ It computes the generalized psi function similarly to the one used in the GN model

        :return: generalized_psi
        """
        f_eval = carrier_cut.frequency

        # Fiber parameters
        alpha0 = self.alpha0(f_eval)
        beta2 = self.fiber_information.beta2
        beta3 = self.fiber_information.beta3
        f_ref_beta = self.fiber_information.f_ref_beta

        z = self.srs_profile.stimulated_raman_scattering.z
        frequency_rho = self.srs_profile.stimulated_raman_scattering.frequency
        rho = self.srs_profile.stimulated_raman_scattering.rho
        rho = rho * np.exp(np.abs(alpha0) * z / 2)
        rho_function = interp1d(frequency_rho, rho, axis=0, fill_value='extrapolate')
        rho_pump = rho_function(pump_carrier.frequency)

        f_resolution = self.model_parameters.frequency_resolution
        f1_array = np.arange(pump_carrier.frequency - pump_carrier.baud_rate,
                             pump_carrier.frequency + pump_carrier.baud_rate,
                             f_resolution)
        f2_array = np.arange(carrier_cut.frequency - carrier_cut.baud_rate,
                                 carrier_cut.frequency + carrier_cut.baud_rate,
                                 f_resolution)
        psd1 = ut.raised_cosine_comb(f1_array, pump_carrier)

        integrand_f1 = np.zeros(len(f1_array))
        for f1_index, (f1, psd1_sample) in enumerate(zip(f1_array, psd1)):
            f3_array = f1 + f2_array - f_eval
            psd2 = ut.raised_cosine_comb(f2_array, carrier_cut)
            psd3 = ut.raised_cosine_comb(f3_array, pump_carrier)
            ggg = psd1_sample * psd2 * psd3

            delta_beta = 4 * np.pi**2 * (f1 - f_eval) * (f2_array - f_eval) * \
                         (beta2 + np.pi * beta3 * (f1 + f2_array - 2 * f_ref_beta))

            # IMPLEMENTATION OF GGN USING delta_rho INSTEAD OF rho_pump
            # delta_rho = rho_function(f1) * rho_function(f2_array) * rho_function(f3_array) / rho_function(f_eval)
            # integrand_f2 = ggg * self._fwm_efficiency(delta_beta, delta_rho, z, alpha0)

            integrand_f2 = ggg * self._generalized_rho_nli(delta_beta, rho_pump, z, alpha0)
            integrand_f1[f1_index] = np.trapz(integrand_f2, f2_array)
        generalized_psi = np.trapz(integrand_f1, f1_array)

        return generalized_psi

    @staticmethod
    def _generalized_rho_nli(delta_beta, rho_pump, z, alpha0):

        w = 1j * delta_beta - alpha0
        generalized_rho_nli = (rho_pump[-1]**2 * np.exp(w * z[-1]) - rho_pump[0]**2 * np.exp(w * z[0])) / w
        for z_ind in range(0, len(z) - 1):
            derivative_rho = (rho_pump[z_ind + 1]**2 - rho_pump[z_ind]**2) / (z[z_ind + 1] - z[z_ind])

            generalized_rho_nli -= derivative_rho * (np.exp(w * z[z_ind + 1]) - np.exp(w * z[z_ind])) / (w ** 2)

        generalized_rho_nli = np.abs(generalized_rho_nli)**2

        return generalized_rho_nli

    # Methods for computing spectrally separated GN
    def _gn_analytic(self, carrier, *carriers):
        """ Computes the nonlinear interference power on a single carrier.
        The method uses eq. 120 from arXiv:1209.0394.
        :param carrier: the signal under analysis
        :param carriers: the full WDM comb
        :return: carrier_nli: the amount of nonlinear interference in W on the under analysis
        """

        alpha = self.alpha0() / 2
        length = self.fiber_information.length
        effective_length = (1 - np.exp(-2 * alpha * length)) / (2 * alpha)
        asymptotic_length = 1 / (2 * alpha)

        beta2 = self.fiber_information.beta2
        gamma = self.fiber_information.gamma

        g_nli = 0
        for interfering_carrier in carriers:
            psi = self._psi(carrier, interfering_carrier)
            g_nli += (interfering_carrier.power.signal / interfering_carrier.baud_rate) ** 2 * \
                     (carrier.power.signal / carrier.baud_rate) * psi

        g_nli *= (16 / 27) * (gamma * effective_length) ** 2 /\
                 (2 * np.pi * abs(beta2) * asymptotic_length)

        carrier_nli = carrier.baud_rate * g_nli
        return carrier_nli

    def _psi(self, carrier, interfering_carrier):
        """ Calculates eq. 123 from arXiv:1209.0394.
        """
        alpha = self.alpha0() / 2
        beta2 = self.fiber_information.beta2

        asymptotic_length = 1 / (2 * alpha)
        if carrier.channel_number == interfering_carrier.channel_number:  # SCI
            psi = np.arcsinh(0.5 * np.pi**2 * asymptotic_length
                              * abs(beta2) * carrier.baud_rate**2)
        else:  # XCI
            delta_f = carrier.frequency - interfering_carrier.frequency
            psi = np.arcsinh(np.pi**2 * asymptotic_length * abs(beta2) *
                             carrier.baud_rate * (delta_f + 0.5 * interfering_carrier.baud_rate))
            psi -= np.arcsinh(np.pi**2 * asymptotic_length * abs(beta2) *
                              carrier.baud_rate * (delta_f - 0.5 * interfering_carrier.baud_rate))

        return psi

    # Methods for computing brute force GGN
    def _compute_ggn_integral(self, carrier, *carriers):

        # Channel under test
        f_eval = carrier.frequency

        # Fiber parameters
        alpha0 = self.alpha0(f_eval)
        beta2 = self.fiber_information.beta2
        beta3 = self.fiber_information.beta3
        f_ref_beta = self.fiber_information.f_ref_beta
        gamma = self.fiber_information.gamma

        z = self.srs_profile.stimulated_raman_scattering.z
        frequency_rho = self.srs_profile.stimulated_raman_scattering.frequency
        rho = self.srs_profile.stimulated_raman_scattering.rho
        rho = rho * np.exp(np.abs(alpha0) * z / 2)

        # PSD generation
        f_resolution = self.model_parameters.frequency_resolution
        start_frequency_psd = min(carriers, key=attrgetter('frequency')).frequency - \
            (min(carriers, key=attrgetter('frequency')).baud_rate / 2)
        stop_frequency_psd = max(carriers, key=attrgetter('frequency')).frequency + \
            (min(carriers, key=attrgetter('frequency')).baud_rate / 2)
        num_samples = int((stop_frequency_psd - start_frequency_psd) / f_resolution) + 1
        frequency_psd = np.array([start_frequency_psd + ii * f_resolution for ii in range(0, num_samples)])
        frequency_psd = np.arange(min(frequency_rho) - f_resolution, max(frequency_rho) + f_resolution, f_resolution)

        psd = ut.raised_cosine_comb(frequency_psd, *carriers)
        f1_array = frequency_psd
        f2_array = frequency_psd
        len_carriers = len(carriers)
        g1 = psd
        g2 = psd

        # Interpolation of SRS gain/loss profile
        rho_function = interp1d(frequency_rho, rho, axis=0, fill_value='extrapolate')
        rho_1 = rho_function(f1_array)
        rho_f = rho_function(f_eval)

        # Progressbar initialization
        if self.model_parameters.verbose:
            bar = progressbar.ProgressBar(maxval=len(f1_array),
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            print(f' NLI computation on channel #{carrier.channel_number}')
            bar.start()

        # NLI computation
        integrand_f1 = np.zeros(f1_array.size)  # pre-allocate partial result for inner integral
        for f_ind, f1 in enumerate(f1_array):  # loop over f1
            if g1[f_ind] == 0:
                continue
            f2_array = self._compute_dense_regimes(f1, f_eval, frequency_psd, len_carriers)
            f3_array = f1 + f2_array - f_eval
            g2 = ut.raised_cosine_comb(f2_array, *carriers)

            g3 = ut.raised_cosine_comb(f3_array, *carriers)
            ggg = g2 * g3 * g1[f_ind]

            if np.count_nonzero(ggg):
                delta_beta = 4 * np.pi ** 2 * (f1 - f_eval) * (f2_array - f_eval) * \
                             (beta2 + np.pi * beta3 * (f1 + f2_array - 2 * f_ref_beta))
                
                rho_2 = rho_function(f2_array)
                rho_3 = rho_function(f3_array)
                delta_rho = rho_1[f_ind, :] * rho_2 * rho_3 / rho_f

                fwm_eff = self._fwm_efficiency(delta_beta, delta_rho, z, alpha0)  # compute FWM efficiency

                integrand_f1[f_ind] = np.trapz(fwm_eff * ggg, f2_array)  # compute inner integral

            if self.model_parameters.verbose:
                bar.update(f_ind)

        gnli = 16.0 / 27.0 * gamma**2 * np.trapz(integrand_f1, f1_array)  # compute outer integral

        carrier_nli = carrier.baud_rate * gnli

        return carrier_nli

    @staticmethod
    def _fwm_efficiency(delta_beta, delta_rho, z, alpha0):
        """ Computes the four-wave mixing efficiency
        """
        w = 1j*delta_beta - alpha0
        fwm_eff = (delta_rho[:, -1] * np.exp(w * z[-1]) - delta_rho[:, 0] * np.exp(w*z[0])) / w
        for z_ind in range(0, len(z) - 1):
            derivative_rho = (delta_rho[:, z_ind + 1] - delta_rho[:, z_ind]) / (z[z_ind + 1] - z[z_ind])

            fwm_eff -= derivative_rho * (np.exp(w * z[z_ind + 1]) - np.exp(w * z[z_ind])) / (w**2)

        fwm_eff = np.abs(fwm_eff) ** 2

        return fwm_eff
