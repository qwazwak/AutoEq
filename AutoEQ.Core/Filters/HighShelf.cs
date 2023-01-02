using AutoEQ.Helper;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AutoEQ.Core.Filters;
public class HighShelf : ShelfFilter
{
    public HighShelf(IEnumerable<double> f, int fs) : base(f, fs)
    {
    }

    /// <summary>
    /// Initializes optimizable center frequency (fc), quality (q) and gain
    /// </summary>
    /// <remarks>
    /// The operating principle is to find a point after which the average level is greatest and set the center
    /// frequency there.Gain is set to average level of the target after the transition band.Quality is always set to 0.7.
    /// </remarks>
    /// <param name="target">Equalizer target frequency response</param>
    /// <param name="fs"></param>
    public override (double? CenterFrequency, double? Quality, double? Gain) init(IList<double> target)
    {
        double? CenterFrequency = null;
        (double? Quality, double? Gain) = Shelfinit(target);
        if (params_fc.HasValue)
        {
            // Find point where the ratio of average level after the point and average level before the point is the greatest
            int min_ix = f.Count(f => f < Math.Max(40, params_fc.Value.min));
            int max_ix = f.Count(f => f < Math.Min(10000, params_fc.Value.max));

            int ix = Enumerable.Range(min_ix, max_ix - min_ix).Select(ix => Math.Abs(target.EnumerateFrom(ix).Average())).argmax();
            fc = Math.Clamp(f[ix], params_fc.Value.min, params_fc.Value.max);
            CenterFrequency = Math.Log10(fc.Value);
        }

        return (CenterFrequency, Quality, Gain);
    }

    /// <summary>
    /// Calculates 2nd order biquad filter coefficients
    /// </summary>
    public override (double a0, double a1, double a2, double b0, double b1, double b2) biquad_coefficients
    {
        get
        {
            double a = Math.Pow(10, gain!.Value / 40);
            double w0 = Math.Tau * fc!.Value / _fs;
            double alpha = Math.Sin(w0) / (2 * q!.Value);

            double a0 = a + 1 - ((a - 1) * Math.Cos(w0)) + (2 * Math.Sqrt(a) * alpha);
            double a1 = -(2 * (a - 1 - ((a + 1) * Math.Cos(w0)))) / a0;
            double a2 = -(a + 1 - ((a - 1) * Math.Cos(w0)) - (2 * Math.Sqrt(a) * alpha)) / a0;

            double b0 = a * (a + 1 + ((a - 1) * Math.Cos(w0)) + (2 * Math.Sqrt(a) * alpha)) / a0;
            double b1 = -2 * a * (a - 1 + ((a + 1) * Math.Cos(w0))) / a0;
            double b2 = a * (a + 1 + ((a - 1) * Math.Cos(w0)) - (2 * Math.Sqrt(a) * alpha)) / a0;

            return (1.0, a1, a2, b0, b1, b2);
        }
    }

}
