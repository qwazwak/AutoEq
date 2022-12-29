using AutoEQ2.Helper;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.Intrinsics.X86;

namespace AutoEQ.Core.Filters;
public abstract class ShelfFilter : PEQFilter
{
    public ShelfFilter(IEnumerable<double> f, int fs) : base(f, fs)
    {
        params_gain = (Constants.DEFAULT_SHELF_FILTER_MIN_GAIN, Constants.DEFAULT_SHELF_FILTER_MAX_GAIN);
        params_fc = (Constants.DEFAULT_SHELF_FILTER_MIN_FC, Constants.DEFAULT_SHELF_FILTER_MAX_FC);
        params_q = (Constants.DEFAULT_SHELF_FILTER_MIN_Q, Constants.DEFAULT_SHELF_FILTER_MAX_Q);
    }

    /// <summary>
    /// Shelf filters start to overshoot hard before they get anywhere near 18 dB per octave slope
    /// </summary>
    public override double sharpness_penalty => 0.0;

        /// <summary>
        /// Calculates penalty for transition band extending Nyquist frequency
        /// </summary>
        /// <remarks>Biquad filter shape starts to get distorted when the transition band extends Nyquist frequency in such a way
        /// that the right side gets compressed(greater slope). This method calculates the MSE between
        /// the left and right sides of the frequency response.If the right side is fully compressed, the penalty is the
        /// entire effect of frequency response thus negating the filter entirely.Right side is mirrored around both axes.</remarks>
        /// <param name=""></param>
        /// <returns></returns>
    public override double band_penalty { get
        {
            //Index to frequency array closes to center frequency
            double fc_ix = np.argmin(np.abs(self.f - self.fc));
            // Number of indexes on each side of center frequency, not extending outside, only up to 10 kHz
            double n = Math.Min(fc_ix, ix10k - fc_ix);
            if (n == 0)
                return 0.0;

            return np.mean(np.square(self.fr[fc_ix - n:fc_ix] - (self.gain - self.fr[fc_ix + n - 1:fc_ix - 1:-1])));
        }
    }
    /// <summary>
    /// Initializes optimizable center frequency (fc), qualtiy (q) and gain
    /// The operating principle is to find a point before which the average level is greatest and set the center
    /// frequency there.Gain is set to average level of the target before the transition band.Quality is always set to 0.7.
    /// </summary>
    /// <param name="target"></param>
    /// <returns></returns>
    /// <exception cref="System.NotImplementedException"></exception>
    protected (double? Quality, double? Gain) Shelfinit(IList<double> target)
    {
        double? Quality = null;
        double? Gain = null;
        if (optimize_q)
            Quality = q = Math.Clamp(0.7, params_q!.Value.min, params_q!.Value.max);

        if (optimize_gain)
        {
            // Calculated weighted average from the target where the frequency response (dBs) of a 1 dB shelf is the weight vector
            gain = 1;
            gain = target.DotProduct(fr) / fr.Sum(); // Weighted average
            Gain = gain = Math.Clamp(gain.Value, params_gain!.Value.min, params_gain!.Value.max);
        }

        return (Quality, Gain);
    }
}
