using System;
using System.Collections.Generic;
using System.Linq;

namespace AutoEQ2.Core;
public abstract class PEQFilter
{
    private bool optimize_fc => params_fc != null;
    private (double min, double max)? params_fc { get; init; }

    private bool optimize_q => params_q != null;
    private (double min, double max)? params_q { get; init; }

    private bool optimize_gain => params_gain != null;
    private (double min, double max)? params_gain { get; init; }

    public PEQFilter(IEnumerable<double> Frequencies, int SampleFreq,
                 double? FreqCenter = null, (double min, double max)? params_fc = null,
                 double? q = null, (double min, double max)? params_q = null,
                 double? gain = null, (double min, double max)? params_gain = null)
    {
        this.Frequencies = Frequencies.ToList();
        this.SampleFreq = SampleFreq;

        this.params_fc = params_fc;
        if (!optimize_fc && FreqCenter == null)
            throw new ArgumentException("fc must be given when not optimizing it");
        _FreqCenter = FreqCenter;

        this.params_q = params_q;
        if (!optimize_q && q == null)
            throw new ArgumentException("q must be given when not optimizing it");
        _q = q;

        this.params_gain = params_gain;
        if (!optimize_gain && gain == null)
            throw new ArgumentException("gain must be given when not optimizing it");
        _gain = gain;
    }

    public override string ToString() => $"{this.GetType().Name} {FreqCenter:.0f} Hz, {q:.2f} Q, {gain:.1f} dB";

    protected List<double>? _f;
    public List<double> Frequencies
    {
        get => _f; set
        {
            _ix10k = null;
            _ix20k = null;
            _fr = null;
            _f = value;
        }
    }

    protected int? _fs;
    public int SampleFreq
    {
        get => _fs; set
        {
            _fr = null;
            _fs = value;
        }
    }
    protected double? _FreqCenter;
    public double FreqCenter
    {
        get => _FreqCenter; set
        {
            _fr = null;
            _FreqCenter = value;
        }
    }
    protected double? _q;
    public double q
    {
        get => _q; set
        {
            _fr = null;
            _q = value;
        }
    }

    protected double? _gain;
    public double gain
    {
        get => _gain; set
        {
            _fr = null;
            _gain = value;
        }
    }

    protected double? _ix10k = null;
    public double ix10k
    {
        get
        {

            if (_ix10k.HasValue)
                return _ix10k.Value;
            
            return Frequencies.Select(f => Math.Abs(f - SampleFreq)).Min();
        }
    }

    protected double? _ix20k = null;
    public double ix20k => _ix20k;

    private IList<double>? _fr = null;
    /// <summary>
    /// Calculates frequency response
    /// </summary>
    public IList<double> fr { get
        {
            if (_fr != null)
                return _fr;
            (double a0, double a1, double a2, double b0, double b1, double b2) = biquad_coefficients;
            a1 *= -1;
            a2 *= -1;

            return (from f in Frequencies
                    let w = Math.Tau * (f / SampleFreq)
                    let phi = 4 * Math.Pow(Math.Sin(w / 2), 2)
                    select (10 * Math.Log10(Math.Pow(b0 + b1 + b2, 2) + (((b0 * b2 * phi) - ((b1 * (b0 + b2)) + (4 * b0 * b2))) * phi)))
                        - (10 * Math.Log10(Math.Pow(a0 + a1 + a2, 2) + (((a0 * a2 * phi) - ((a1 * (a0 + a2)) + (4 * a0 * a2))) * phi)))).ToList();
        }
    }
    /// <summary>
    /// Initializes optimizable center frequency (fc), qualtiy (q) and gain
    /// </summary>
    /// <param name="target">Equalizer target frequency response</param>
    /// <returns>List of initialized optimizable parameter values for the optimizer</returns>
    public abstract dynamic init(dynamic target);

    public abstract (double a0, double a1, double a2, double b0, double b1, double b2) biquad_coefficients { get; }
    public abstract double sharpness_penalty { get; }
    public abstract double band_penalty { get; }
}