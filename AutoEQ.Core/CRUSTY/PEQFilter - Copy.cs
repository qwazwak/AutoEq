/*
 * 
 * import warnings
from time import time
from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt, ticker
from scipy.optimize import fmin_slsqp
from scipy.signal import find_peaks
from tabulate import tabulate

from autoeq.constants import Constants.DEFAULT_SHELF_FILTER_MIN_FC, Constants.DEFAULT_SHELF_FILTER_MAX_FC, Constants.DEFAULT_SHELF_FILTER_MIN_Q, \
    Constants.DEFAULT_SHELF_FILTER_MAX_Q, Constants.DEFAULT_SHELF_FILTER_MIN_GAIN, Constants.DEFAULT_SHELF_FILTER_MAX_GAIN, \
    Constants.DEFAULT_PEAKING_FILTER_MIN_FC, Constants.DEFAULT_PEAKING_FILTER_MAX_FC, Constants.DEFAULT_PEAKING_FILTER_MIN_Q, \
    Constants.DEFAULT_PEAKING_FILTER_MAX_Q, Constants.DEFAULT_PEAKING_FILTER_MIN_GAIN, Constants.DEFAULT_PEAKING_FILTER_MAX_GAIN, \
    Constants.DEFAULT_PEQ_OPTIMIZER_MIN_F, Constants.DEFAULT_PEQ_OPTIMIZER_MAX_F, Constants.DEFAULT_PEQ_OPTIMIZER_MAX_TIME, \
    Constants.DEFAULT_PEQ_OPTIMIZER_TARGET_LOSS, Constants.DEFAULT_PEQ_OPTIMIZER_MIN_CHANGE_RATE, Constants.DEFAULT_PEQ_OPTIMIZER_MIN_STD
*/

using System.Collections.Generic;
using System.Linq;

namespace AutoEQ2.Core;
public class Peaking : PEQFilter
{
    public Peaking(IEnumerable<double> f, object fs,
                 double? fc = null, bool? optimize_fc = null, double? min_fc = null, double? max_fc = null,
                 double? q = null, bool? optimize_q = null, double? min_q = null, double? max_q = null,
                 double? gain = null, bool? optimize_gain = null, double? min_gain = null, double? max_gain = null) : base(f, fs, fc, optimize_fc, min_fc, max_fc, q, optimize_q, min_q, max_q, gain, optimize_gain, min_gain, max_gain)
    {
    }
    /// <summary>
    /// Initializes optimizable center frequency (fc), qualtiy (q) and gain
    /// The operating principle is to find the biggest(by width AND height) peak of the target curve and set center
    /// frequency at the peak's location. Quality is set in such a way that the filter width matches the peak width
    /// and gain is set to the peak height.
    /// </summary>
    /// <param name="target">Equalizer target frequency response</param>
    /// <returns>List of initialized optimizable parameter values for the optimizer</returns>
    /// <exception cref="System.NotImplementedException"></exception>
    public override dynamic init(dynamic target)
    {
        // Finds positive and negative peaks
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            positive_peak_ixs, peak_props = find_peaks(np.clip(target, 0, None), width = 0, prominence = 0, height = 0)
            negative_peak_ixs, dip_props = find_peaks(np.clip(-target, 0, None), width = 0, prominence = 0, height = 0)

        // Indexes for minimum and maximum center frequency
        min_fc_ix = np.argmin(np.abs(self.f - self.min_fc))
        max_fc_ix = np.argmin(np.abs(self.f - self.max_fc))

        if(len(positive_peak_ixs) == 0 and len(negative_peak_ixs) == 0)
            // No peaks found
            params = []
            if(self.optimize_fc)
                self.fc = self.f[(min_fc_ix + max_fc_ix) // 2]
                params.append(np.log10(self.fc))
            if(self.optimize_q)
                self.q = np.sqrt(2)
                params.append(self.q)
            if(self.optimize_gain)
                self.gain = 0.0
                params.append(self.gain)
            return params

        // All peak indexes together
        peak_ixs = np.concatenate([positive_peak_ixs, negative_peak_ixs])
        // Exclude peak indexes which are outside of minimum and maximum center frequency
        mask = np.logical_and(peak_ixs >= min_fc_ix, peak_ixs <= max_fc_ix)
        peak_ixs = peak_ixs[mask]
        // Properties of included peaks together
        widths = np.concatenate([peak_props["widths"], dip_props["widths"]])[mask]
        heights = np.concatenate([peak_props["peak_heights"], dip_props["peak_heights"]])[mask]
        // Find the biggest peak, by height AND width
        sizes = widths * heights  # Size of each peak for ranking
        ixs_ix = np.argmax(sizes)  # Index to indexes array which point to the biggest peak
        ix = peak_ixs[ixs_ix]  # Index to f and target

        params = []
        if(self.optimize_fc)
            self.fc = np.clip(self.f[ix], self.min_fc, self.max_fc)
            params.append(np.log10(self.fc))  # Convert to logarithmic scale for optimizer
        if(self.optimize_q)
            width = widths[ixs_ix]
            // Find bandwidth which matches the peak width
            f_step = np.log2(self.f[1] / self.f[0])
            bw = np.log2((2 * *f_step) * *width)
            // Calculate quality with bandwidth
            self.q = np.sqrt(2 * *bw) / (2 * *bw - 1)
            self.q = np.clip(self.q, self.min_q, self.max_q)
            params.append(self.q)
        if(self.optimize_gain)
            // Target value at center frequency
            self.gain = heights[ixs_ix] if target[ix] > 0 else -heights[ixs_ix]
            self.gain = np.clip(self.gain, self.min_gain, self.max_gain)
            params.append(self.gain)
        return params
   }

    def biquad_coefficients(self):
        """Calculates 2nd order biquad filter coefficients"""
        a = 10 * *(self.gain / 40)
        w0 = 2 * np.pi * self.fc / self._fs
        alpha = np.sin(w0) / (2 * self.q)

        a0 = 1 + alpha / a
        a1 = -(-2 * np.cos(w0)) / a0
        a2 = -(1 - alpha / a) / a0

        b0 = (1 + alpha * a) / a0
        b1 = (-2 * np.cos(w0)) / a0
        b2 = (1 - alpha * a) / a0

        return 1.0, a1, a2, b0, b1, b2

    @property
    def sharpness_penalty(self):
        """Calculates penalty for having too steep slope

        Multiplies the filter frequency response with a penalty coefficient and calculates MSE from it

        The penalty coefficient is a sigmoid function which goes quickly from 0.0 to 1.0 around 18 dB / octave slope
        """
        // This polynomial function gives the gain for peaking filter which achieves 18 dB / octave max derivative
        // The polynomial estimate is accurate in the vicinity of 18 dB / octave
        gain_limit = -0.09503189270199464 + 20.575128011847003 * (1 / self.q)
        // Scaled sigmoid function as penalty coefficient
        x = self.gain / gain_limit - 1
        sharpness_penalty_coefficient = 1 / (1 + np.e * *(-x * 100))
        return np.mean(np.square(self.fr* sharpness_penalty_coefficient))

    @property
    def band_penalty(self) :
        """Calculates penalty for transition band extending Nyquist frequency

        Biquad filter shape starts to get distorted when the transition band extends Nyquist frequency in such a way
        that the right side gets compressed(greater slope). This method calculates the RMSE between
        the left and right sides of the frequency response.If the right side is fully compressed, the penalty is the
       entire effect of frequency response thus negating the filter entirely.Right side is mirrored around vertical
      axis.
        """
        fc_ix = np.argmin(np.abs(self.f - self.fc))  # Index to frequency array closes to center frequency
        // Number of indexes on each side of center frequency, not extending outside, only up to 10 kHz

      n = min(fc_ix, self.ix10k - fc_ix)
        if(n == 0)
            return 0.0
        return np.mean(np.square(self.fr[fc_ix - n:fc_ix] - self.fr[fc_ix + n - 1:fc_ix - 1:-1]))

        }
class ShelfFilter : PEQFilter
{
    def __init__(self, f, fs, fc= None, optimize_fc= None, min_fc= Constants.DEFAULT_SHELF_FILTER_MIN_FC,
                 max_fc= Constants.DEFAULT_SHELF_FILTER_MAX_FC, q= None, optimize_q= None, min_q= Constants.DEFAULT_SHELF_FILTER_MIN_Q,
                 max_q= Constants.DEFAULT_SHELF_FILTER_MAX_Q, gain= None, optimize_gain= None,
                 min_gain= Constants.DEFAULT_SHELF_FILTER_MIN_GAIN, max_gain= Constants.DEFAULT_SHELF_FILTER_MAX_GAIN):
        super().__init__(f, fs, fc, optimize_fc, min_fc, max_fc, q, optimize_q, min_q, max_q, gain, optimize_gain,
                         min_gain, max_gain)

    @property
    def sharpness_penalty(self) :
        // Shelf filters start to overshoot hard before they get anywhere near 18 dB per octave slope
        return 0.0

    @property
    def band_penalty(self) :
        """Calculates penalty for transition band extending Nyquist frequency

        Biquad filter shape starts to get distorted when the transition band extends Nyquist frequency in such a way
        that the right side gets compressed(greater slope). This method calculates the MSE between
        the left and right sides of the frequency response.If the right side is fully compressed, the penalty is the
       entire effect of frequency response thus negating the filter entirely.Right side is mirrored around both axes.
        """
        fc_ix = np.argmin(np.abs(self.f - self.fc))  # Index to frequency array closes to center frequency
        // Number of indexes on each side of center frequency, not extending outside, only up to 10 kHz

      n = min(fc_ix, self.ix10k - fc_ix)
        if(n == 0)
            return 0.0
        return np.mean(np.square(self.fr[fc_ix - n:fc_ix] - (self.gain - self.fr[fc_ix + n - 1:fc_ix - 1:-1])))
        }

public class HighShelf : ShelfFilter
    def init(self, target):
        """Initializes optimizable center frequency (fc), quality (q) and gain

        The operating principle is to find a point after which the average level is greatest and set the center
        frequency there. Gain is set to average level of the target after the transition band. Quality is always set to
        0.7.

        Args:
            target: Equalizer target frequency response

        Returns:
            List of initialized optimizable parameter values for the optimizer
        """
        params = []
        if(self.optimize_fc)
            // Find point where the ratio of average level after the point and average level before the point is the
            // greatest
            min_ix = np.sum(self.f < max(40, self.min_fc))
            max_ix = np.sum(self.f < min(10000, self.max_fc))
            ix = np.argmax([np.abs(np.mean(target[ix:])) for ix in range(min_ix, max_ix)])
            self.fc = np.clip(self.f[ix], self.min_fc, self.max_fc)
            params.append(np.log10(self.fc))
        if(self.optimize_q)
            self.q = np.clip(0.7, self.min_q, self.max_q)
            params.append(self.q)
        if(self.optimize_gain)
            // Calculated weighted average from the target where the frequency response (dBs) of a 1 dB shelf is the
            // weight vector
            self.gain = 1
            self.gain = np.dot(target, self.fr) / np.sum(self.fr)  # Weighted average
            self.gain = np.clip(self.gain, self.min_gain, self.max_gain)
            params.append(self.gain)
        return params

    def biquad_coefficients(self):
        """Calculates 2nd order biquad filter coefficients"""
        a = 10 ** (self.gain / 40)
        w0 = 2 * np.pi * self.fc / self._fs
        alpha = np.sin(w0) / (2 * self.q)

        a0 = (a + 1) - (a - 1) * np.cos(w0) + 2 * np.sqrt(a) * alpha
        a1 = -(2 * ((a - 1) - (a + 1) * np.cos(w0))) / a0
        a2 = -((a + 1) - (a - 1) * np.cos(w0) - 2 * np.sqrt(a) * alpha) / a0

        b0 = (a * ((a + 1) + (a - 1) * np.cos(w0) + 2 * np.sqrt(a) * alpha)) / a0
        b1 = (-2 * a * ((a - 1) + (a + 1) * np.cos(w0))) / a0
        b2 = (a * ((a + 1) + (a - 1) * np.cos(w0) - 2 * np.sqrt(a) * alpha)) / a0

        return 1.0, a1, a2, b0, b1, b2


class LowShelf(ShelfFilter):
    def init(self, target):
        """Initializes optimizable center frequency (fc), qualtiy (q) and gain

        The operating principle is to find a point before which the average level is greatest and set the center
        frequency there. Gain is set to average level of the target before the transition band. Quality is always set to
        0.7.

        Args:
            target: Equalizer target frequency response

        Returns:
            List of initialized optimizable parameter values for the optimizer
        """
        params = []
        if(self.optimize_fc)
            // Find point where the ratio of average level before the point and average level after the point is the
            // greatest
            min_ix = np.sum(self.f < max(40, self.min_fc))
            max_ix = np.sum(self.f < min(10000, self.max_fc))
            ix = np.argmax([np.abs(np.mean(target[:ix + 1])) for ix in range(min_ix, max_ix)])
            ix += min_ix
            self.fc = np.clip(self.f[ix], self.min_fc, self.max_fc)
            params.append(np.log10(self.fc))
        if(self.optimize_q)
            self.q = np.clip(0.7, self.min_q, self.max_q)
            params.append(self.q)
        if(self.optimize_gain)
            // Calculated weighted average from the target where the frequency response (dBs) of a 1 dB shelf is the
            // weight vector
            self.gain = 1
            self.gain = np.dot(target, self.fr) / np.sum(self.fr)  # Weighted average
            self.gain = np.clip(self.gain, self.min_gain, self.max_gain)
            params.append(self.gain)
        return params

    def biquad_coefficients(self):
        """Calculates 2nd order biquad filter coefficients"""
        a = 10 ** (self.gain / 40)
        w0 = 2 * np.pi * self.fc / self._fs
        alpha = np.sin(w0) / (2 * self.q)

        a0 = (a + 1) + (a - 1) * np.cos(w0) + 2 * np.sqrt(a) * alpha
        a1 = -(-2 * ((a - 1) + (a + 1) * np.cos(w0))) / a0
        a2 = -((a + 1) + (a - 1) * np.cos(w0) - 2 * np.sqrt(a) * alpha) / a0

        b0 = (a * ((a + 1) - (a - 1) * np.cos(w0) + 2 * np.sqrt(a) * alpha)) / a0
        b1 = (2 * a * ((a - 1) - (a + 1) * np.cos(w0))) / a0
        b2 = (a * ((a + 1) - (a - 1) * np.cos(w0) - 2 * np.sqrt(a) * alpha)) / a0

        return 1.0, a1, a2, b0, b1, b2

