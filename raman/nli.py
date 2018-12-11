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


    def compute_dense_regimes(self, f1, f_eval,frequency_psd,len_carriers,alpha0,beta2):

        f_central = min(frequency_psd) + (max(frequency_psd) - min(frequency_psd)) / 2
        frequency_psd = frequency_psd - f_central
        f_eval = f_eval - f_central
        f1 = f1 - f_central

        NpointsPerSlotMin = 4
        NpointsPerSlotMax = 5000 / 5

        Deltaf = 50e9

        DeltafMin = Deltaf / NpointsPerSlotMax
        DeltafMax = Deltaf / NpointsPerSlotMin
        Bopt = max([len_carriers * Deltaf, max(frequency_psd) - min(frequency_psd) + Deltaf])
        fMax = 0.6 * Bopt
        minFWMinv = 1E6
        alpha_e = (alpha0 / 2)


        f2DenseUpLimit = max(
            [f_eval + np.sqrt(alpha_e ** 2 / (4 * (np.pi ** 4) * (beta2 ** 2)) * (minFWMinv - 1)) / (f1 - f_eval)  ,
             f_eval - np.sqrt(alpha_e ** 2 / (4 * (np.pi ** 4) * (beta2 ** 2)) * (minFWMinv - 1)) / (f1 - f_eval)])     # Limit on f2 based on classic FWM
        f2DenseLowLimit = min(
            [f_eval + np.sqrt(alpha_e ** 2 / (4 * (np.pi ** 4) * (beta2 ** 2)) * (minFWMinv - 1)) / (f1 - f_eval),
             f_eval - np.sqrt(alpha_e ** 2 / (4 * (np.pi ** 4) * (beta2 ** 2)) * (minFWMinv - 1)) / (f1 - f_eval)])

        if f2DenseLowLimit == 0:
            f2DenseLowLimit = -DeltafMin

        if f2DenseUpLimit == 0:
            f2DenseUpLimit = DeltafMin

        if f2DenseLowLimit < -fMax:
            f2DenseLowLimit = -fMax

        if f2DenseUpLimit > fMax:
            f2DenseUpLimit = fMax

        f2DenseWidth = abs(f2DenseUpLimit - f2DenseLowLimit)
        NpointsDense = np.ceil(
            f2DenseWidth / DeltafMin)  # Number of integration points to be considered in the denser area
        if NpointsDense < 100:
            NpointsDense = 100

        DeltafArray = f2DenseWidth / NpointsDense  # Get frequency spacing
        f2ArrayDense = np.arange(f2DenseLowLimit, f2DenseUpLimit, DeltafArray)  # Define the denser grid

        if f_eval < 0:
            k = Bopt / 2 / (Bopt / 2 - DeltafMax)  # Get step ratio for logspace array definition
            NlogShort = np.ceil(np.log(fMax / abs(f2DenseLowLimit)) * 1 / np.log(
                k) + 1)  # Get number of points required to ensure that the maximum frequency step in Bopt is not passed
            f2Short = -(abs(f2DenseLowLimit) * k ** (np.arange(NlogShort, 0, -1) - 1))  # Generate logspace array
            k = (Bopt / 2 + (abs(f2DenseUpLimit) - f2DenseUpLimit)) / (
                    Bopt / 2 - DeltafMax + (abs(f2DenseUpLimit) - f2DenseUpLimit))
            NlogLong = np.ceil(
                np.log((fMax + (abs(f2DenseUpLimit) - f2DenseUpLimit)) / abs(f2DenseUpLimit)) * 1 / np.log(
                    k) + 1)  # Get number of points required to ensure that the maximum frequency step in Bopt is not passed
            f2Long = (abs(f2DenseUpLimit) * k ** (np.arange(1, NlogLong + 1, 1) - 1)) - (
                    abs(f2DenseUpLimit) - f2DenseUpLimit)
            f2Array = np.array(list(f2Short) + list(f2ArrayDense[1:]) + list(f2Long))
        else:
            k = Bopt / 2 / (Bopt / 2 - DeltafMax)
            NlogShort = np.ceil(np.log(fMax / abs(f2DenseUpLimit)) * 1 / np.log(k) + 1)
            f2Short = f2DenseUpLimit * k ** (np.arange(1, NlogShort + 1, 1) - 1)
            k = (Bopt / 2 + (abs(f2DenseLowLimit) + f2DenseLowLimit)) / (
                    Bopt / 2 - DeltafMax + (abs(f2DenseLowLimit) + f2DenseLowLimit))
            NlogLong = np.ceil(
                np.log((fMax + (abs(f2DenseLowLimit) + f2DenseLowLimit)) / abs(f2DenseLowLimit)) * 1 / np.log(
                    k) + 1)
            f2Long = -(abs(f2DenseLowLimit) * k ** (np.arange(NlogLong, 0, -1) - 1)) + (
                    abs(f2DenseLowLimit) + f2DenseLowLimit)
            f2Array = np.array(list(f2Long) + list(f2ArrayDense[1:]) + list(f2Short))

        return f2Array + f_central

    def compute_nli(self, carrier, *carriers):
        """
        Compute NLI power generated by the WDM comb `*carriers` on the channel under test `carrier` 
        at the end of the fiber span.
        """
        if self.model_parameters.method.lower() == 'ggn_integral':
            carrier_nli = self._compute_ggn_integral(carrier, *carriers)
        else:
            raise ValueError(f'Method {self.model_parameters.method_nli} not implemented.')

        return carrier_nli

    def _compute_ggn_integral(self, carrier, *carriers):

        # Verify if SRS profile is associated to SRS
        if len(carriers) != len(self.srs_profile.spectral_information.carriers):
            raise ValueError('Number of carriers of `self.srs_profile` is {}, '
                             'while number of carriers in `carriers` is {}. '
                             'They must be the same'.format((len(self.srs_profile.spectral_information.carriers)),
                                                            len(carriers)))

        for index, srs_carrier in enumerate(self.srs_profile.spectral_information.carriers):
            if (srs_carrier.power.signal != carriers[index].power.signal) or \
             (srs_carrier.frequency != carriers[index].frequency):
                raise ValueError('Carrier #{} of self.srs_profile does not match '
                                 'with #{} carrier in *carriers'.format(carriers[index].channel_number,
                                                                        srs_carrier.channel_number))
        # Channel under test
        f_eval = carrier.frequency

        # Fiber parameters
        alpha0 = self.fiber_information.attenuation_coefficient.alpha_power
        beta2 = self.fiber_information.beta2
        beta3 = self.fiber_information.beta3
        gamma = self.fiber_information.gamma

        if len(self.fiber_information.attenuation_coefficient.alpha_power) == 1:
            alpha0 = self.fiber_information.attenuation_coefficient.alpha_power[0]
        else:
            alpha_interp = interp1d(self.fiber_information.attenuation_coefficient.frequency,
                                    self.fiber_information.attenuation_coefficient.alpha_power)
            alpha0 = alpha_interp(f_eval)

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
        frequency_psd = np.arange(min(frequency_rho),max(frequency_rho),f_resolution)
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
            print(' NLI computation on channel #{}'.format(carrier.channel_number))
            bar.start()

        # NLI computation
        integrand_f1 = np.zeros(f1_array.size)  # pre-allocate partial result for inner integral
        for f_ind, f1 in enumerate(f1_array):  # loop over f1
            if g1[f_ind] == 0:
                continue
            f2_array = self.compute_dense_regimes(f1,f_eval,frequency_psd,len_carriers,alpha0,beta2)
            f3_array = f1 + f2_array - f_eval
            g2 = ut.raised_cosine_comb(f2_array, *carriers)
            g3 = ut.raised_cosine_comb(f3_array, *carriers)
            ggg = g2 * g3 * g1[f_ind]


            if np.count_nonzero(ggg):
                delta_beta = 4 * np.pi ** 2 * (f1 - f_eval) * (f2_array - f_eval) * \
                             (beta2 + np.pi * beta3 * (f1 + f2_array))
                rho_2 = rho_function(f2_array)
                rho_3 = rho_function(f3_array)
                delta_rho = rho_1[f_ind, :] * rho_2 * rho_3 / rho_f


                fwm_eff = self._fwm_efficiency(delta_beta, delta_rho, z, alpha0)  # compute FWM efficiency

                integrand_f1[f_ind] = np.trapz(fwm_eff * ggg, f2_array)  # compute inner integral

            if self.model_parameters.verbose:
                bar.update(f_ind)

        gnli = 16.0 / 27.0 * gamma ** 2 * rho_f[-1] ** 2 * np.exp(-np.abs(alpha0) * z[-1]) * \
               np.trapz(integrand_f1, f1_array)  # compute outer integral

        carrier_nli = carrier.baud_rate * gnli

        return carrier_nli

    @staticmethod
    def _fwm_efficiency(delta_beta, delta_rho, z, alpha0):
        """ Computes the four-wave mixing efficiency
        """
        w = 1j*delta_beta - alpha0
        fwm_eff = (delta_rho[:,-1]*np.exp(w*z[-1])-delta_rho[:,0]*np.exp(w*z[0]))/w
        for z_ind in range(0, len(z) - 1):
            derivative_rho = (delta_rho[:, z_ind + 1] - delta_rho[:, z_ind]) / (z[z_ind + 1] - z[z_ind])
            fwm_eff -= derivative_rho * (np.exp(w*z[z_ind +1])-np.exp(w*z[z_ind]))/(w**2)
        fwm_eff = np.abs(fwm_eff) ** 2

        return fwm_eff
