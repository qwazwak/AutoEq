﻿using AutoEQ.Helper;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AutoEQ.Core.Filters;
public abstract class PEQFilter
{
    protected readonly ResettableLazy<int> _ix10k;
    protected readonly ResettableLazy<int> _ix20k;
    public int ix10k => _ix10k.Value;
    public int ix20k => _ix20k.Value;
    private List<double> _f;
    public List<double> f
    {
        get => _f;
        set
        {
            _ix10k.Reset();
            _ix20k.Reset();
            _fr = null;
            _f = value;
        }
    }

    protected int _fs;
    public int fs
    {
        get => _fs; set
        {
            _fr = null;
            _fs = value;
        }
    }

    public (double min, double max)? params_fc { get; init; }
    public (double min, double max)? params_q { get; init; }
    public (double min, double max)? params_gain { get; init; }

    protected double? _fc;
    /// <summary>
    /// Frequency Center
    /// </summary>
    public double? fc
    {
        get => _fc; set
        {
            _fr = null;
            _fc = value;
        }
    }
    protected double? _q;
    public double? q
    {
        get => _q; set
        {
            _fr = null;
            _q = value;
        }
    }

    protected double? _gain;
    public double? gain
    {
        get => _gain; set
        {
            _fr = null;
            _gain = value;
        }
    }

    public PEQFilter(IEnumerable<double> f, int fs)
    {
        _ix10k = new(() => f.IndexClosestTo(10000));
        _ix20k = new(() => f.IndexClosestTo(20000));
        this.f = f.ToList();
        this.fs = fs;
            /*

        if not optimize_fc and fc is None:
            raise TypeError('fc must be given when not optimizing it')
            self._fc = fc
        self.optimize_fc = optimize_fc
        self.min_fc = min_fc
        self.max_fc = max_fc

        if not optimize_fc and fc is None:
            raise TypeError('q must be given when not optimizing it')
            self._q = q
        self.optimize_q = optimize_q
        self.min_q = min_q
        self.max_q = max_q

        if not optimize_fc and fc is None:
            raise TypeError('gain must be given when not optimizing it')
            self._gain = gain
        self.optimize_gain = optimize_gain
        self.min_gain = min_gain
        self.max_gain = max_gain

        self._ix10k = None
        self._ix20k = null;

        self._fr = None
            */
            VerifyState();
    }


    private void VerifyState()
    {
        if(!params_fc.HasValue && fc == null)
            throw new ArgumentException("fc must be given when not optimizing it");
        if (!params_q.HasValue && q == null)
            throw new ArgumentException("q must be given when not optimizing it");
        if (!params_fc.HasValue && fc == null)
           throw new ArgumentException("gain must be given when not optimizing it");
    }
    public override string ToString() => $"{GetType().Name} {fc:.0f} Hz, {q:.2f} Q, {gain:.1f} dB";


    private List<double>? _fr;
    /// <summary>
    /// Calculates frequency response
    /// </summary>
    public IList<double> fr
    {
        get
        {
            if (_fr != null)
                return _fr;
            /*
            IEnumerable<double> w = f.Select(f => Math.Tau * f / fs);
            var phi = w.Select(w => 4 * Math.Pow(Math.Sin(w / 2), 2));

            (double a0, double a1, double a2, double b0, double b1, double b2) = biquad_coefficients;
            a1 *= -1;
            a2 *= -1;

            _fr =
                10 * Math.Log10(Math.Pow((b0 + b1 + b2), 2) + (b0 * b2 * phi - (b1 * (b0 + b2) + 4 * b0 * b2)) * phi)
              - 10 * Math.Log10(Math.Pow((a0 + a1 + a2), 2) + (a0 * a2 * phi - (a1 * (a0 + a2) + 4 * a0 * a2)) * phi);
            return _fr;*/

                (double a0, double a1, double a2, double b0, double b1, double b2) = biquad_coefficients;
                a1 *= -1;
                a2 *= -1;
                double BSquare = Math.Pow(b0 + b1 + b2, 2);
                double ASquare = Math.Pow(a0 + a1 + a2, 2);
                double BAdd = (b1 * (b0 + b2)) + (4 * b0 * b2);
                double AAdd = (a1 * (a0 + a2)) + (4 * a0 * a2);

                double b02 = b0 * b2;
                double a02 = a0 * a2;

                return _fr = f.Select(f =>
                {
                    double w = Math.Tau * f / fs;
                    double phi = 4 * Math.Pow(Math.Sin(w / 2), 2);

                    return 10 *
                        (Math.Log10(BSquare + (((b02 * phi) - BAdd) * phi))
                       - Math.Log10(ASquare + (((a02 * phi) - AAdd) * phi)));
                }).ToList();
        }
    }

    /// <summary>
    /// Initializes optimizable center frequency (fc), qualtiy (q) and gain
    /// </summary>
    /// <param name="target">Equalizer target frequency response</param>
    /// <returns>List of initialized optimizable parameter values for the optimizer</returns>
    public abstract (double? CenterFrequency, double? Quality, double? Gain) init(IList<double> target);

    public abstract (double a0, double a1, double a2, double b0, double b1, double b2) biquad_coefficients { get; }
    public abstract double sharpness_penalty { get; }
    public virtual double band_penalty
    {
        get
        {

            //Index to frequency array closes to center frequency
            //f.Select(i => Math.Abs(i - fc!.Value));
            int fc_ix = f.IndexClosestTo(fc.Value);
            // Number of indexes on each side of center frequency, not extending outside, only up to 10 kHz
            int n = Math.Min(fc_ix, ix10k - fc_ix);
            if (n == 0)
                return 0.0;
            var Lows = fr.EnumerateRange(fc_ix - n, fc_ix, false);
            var highs = fr.EnumerateRangeReverse(fc_ix + n, fc_ix, false);
            return Lows.Zip(highs, (l, h) => Math.Pow(l - h, 2)).Average();
        }
    }
}