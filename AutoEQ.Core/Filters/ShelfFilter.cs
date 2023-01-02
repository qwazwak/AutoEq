using AutoEQ.Helper;
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
        if (optimize_q.HasValue)
            Quality = q = Math.Clamp(0.7, params_q!.Value.min, params_q!.Value.max);

        if (optimize_gain.HasValue)
        {
            // Calculated weighted average from the target where the frequency response (dBs) of a 1 dB shelf is the weight vector
            gain = 1;
            gain = target.DotProduct(fr) / fr.Sum(); // Weighted average
            Gain = gain = Math.Clamp(gain.Value, params_gain!.Value.min, params_gain!.Value.max);
        }

        return (Quality, Gain);
    }
}
