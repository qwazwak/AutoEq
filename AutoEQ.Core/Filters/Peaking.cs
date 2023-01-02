using AutoEQ.Helper;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AutoEQ.Core.Filters;

public class Peaking : PEQFilter
{
    public Peaking(IEnumerable<double> f, int fs) : base(f, fs)
    {
        params_gain = (Constants.DEFAULT_PEAKING_FILTER_MIN_GAIN, Constants.DEFAULT_PEAKING_FILTER_MAX_GAIN);
        params_fc = (Constants.DEFAULT_PEAKING_FILTER_MIN_FC, Constants.DEFAULT_PEAKING_FILTER_MAX_FC);
        params_q = (Constants.DEFAULT_PEAKING_FILTER_MIN_Q, Constants.DEFAULT_PEAKING_FILTER_MAX_Q);
    }
    /// <inheritdoc/>
    /// <remarks>
    /// The operating principle is to find the biggest(by width AND height) peak of the target curve and set center
    /// frequency at the peak's location. Quality is set in such a way that the filter width matches the peak width
    /// and gain is set to the peak height.
    /// </remarks>
    /// <param name="target">Equalizer target frequency response</param>
    /// <returns>List of initialized optimizable parameter values for the optimizer</returns>
    /// <exception cref="System.NotImplementedException"></exception>
    public override (double? CenterFrequency, double? Quality, double? Gain) init(IList<double> target)
    {
        /*
        // Finds positive and negative peaks
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            positive_peak_ixs, peak_props = find_peaks(np.clip(target, 0, None), width = 0, prominence = 0, height = 0)
            negative_peak_ixs, dip_props = find_peaks(np.clip(-target, 0, None), width = 0, prominence = 0, height = 0)

        // Indexes for minimum and maximum center frequency
        min_fc_ix = np.argmin(np.abs(self.f - self.min_fc))
        max_fc_ix = np.argmin(np.abs(self.f - self.max_fc))

        if len(positive_peak_ixs) == 0 and len(negative_peak_ixs) == 0:
            // No peaks found
            params = []
            if self.optimize_fc:
                self.fc = self.f[(min_fc_ix + max_fc_ix) // 2]
                params.append(np.log10(self.fc))
            if self.optimize_q:
                self.q = np.sqrt(2)
                params.append(self.q)
            if self.optimize_gain:
                self.gain = 0.0
                params.append(self.gain)
            return params

        // All peak indexes together
        peak_ixs = np.concatenate([positive_peak_ixs, negative_peak_ixs])
        // Exclude peak indexes which are outside of minimum and maximum center frequency
        mask = np.logical_and(peak_ixs >= min_fc_ix, peak_ixs <= max_fc_ix)
        peak_ixs = peak_ixs[mask]
        // Properties of included peaks together
        widths = np.concatenate([peak_props['widths'], dip_props['widths']])[mask]
        heights = np.concatenate([peak_props['peak_heights'], dip_props['peak_heights']])[mask]
# Find the biggest peak, by height AND width
        sizes = widths * heights  # Size of each peak for ranking
        ixs_ix = np.argmax(sizes)  # Index to indexes array which point to the biggest peak
        ix = peak_ixs[ixs_ix]  # Index to f and target

        params = []
        if self.optimize_fc:
            self.fc = np.clip(self.f[ix], self.min_fc, self.max_fc)
            params.append(np.log10(self.fc))  # Convert to logarithmic scale for optimizer
        if self.optimize_q:
            width = widths[ixs_ix]
            // Find bandwidth which matches the peak width
            f_step = np.log2(self.f[1] / self.f[0])
            bw = np.log2((2 * *f_step) * *width)
            // Calculate quality with bandwidth
            self.q = np.sqrt(2 * *bw) / (2 * *bw - 1)
            self.q = np.clip(self.q, self.min_q, self.max_q)
            params.append(self.q)
        if self.optimize_gain:
            // Target value at center frequency
            self.gain = heights[ixs_ix] if target[ix] > 0 else -heights[ixs_ix]
            self.gain = np.clip(self.gain, self.min_gain, self.max_gain)
            params.append(self.gain)*/
       // return @params;
    }
    // Calculates 2nd order biquad filter coefficients
    public override (double a0, double a1, double a2, double b0, double b1, double b2) biquad_coefficients
    {
        get
        {
            double a = Math.Pow(10, gain.Value / 40);
            double w0 = Math.Tau * fc.Value / _fs;

            double alpha = Math.Sin(w0) / (2 * q.Value);

            double a0 = 1 + alpha / a;
            double a1 = -(-2 * Math.Cos(w0)) / a0;
            double a2 = -(1 - (alpha / a)) / a0;

            double b0 = (1 + (alpha * a)) / a0;
            double b1 = -2 * Math.Cos(w0) / a0;
            double b2 = (1 - (alpha * a)) / a0;

            return (1.0, a1, a2, b0, b1, b2);
        }
    }

    /// <summary>
    /// Calculates penalty for having too steep slope
    /// Multiplies the filter frequency response with a penalty coefficient and calculates MSE from it
    /// The penalty coefficient is a sigmoid function which goes quickly from 0.0 to 1.0 around 18 dB / octave slope
    /// </summary>
    public override double sharpness_penalty
    {
        get
        {
            // This polynomial function gives the gain for peaking filter which achieves 18 dB / octave max derivative
            // The polynomial estimate is accurate in the vicinity of 18 dB / octave
            double gain_limit = -0.09503189270199464 + 20.575128011847003 * (1 / q.Value);
            // Scaled sigmoid function as penalty coefficient
            double x = gain.Value / gain_limit - 1;
            double sharpness_penalty_coefficient = 1 / (1 + Math.Exp(-x * 100));
            return fr.Select(EachResponseFunc).Average();

            double EachResponseFunc(double response) => Math.Pow(response * sharpness_penalty_coefficient, 2);
        }
    }
}