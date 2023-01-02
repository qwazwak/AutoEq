using AutoEQ.Core.Filters;
using AutoEQ.Core.Models;
using AutoEQ.Helper;
using MathNet.Numerics.Optimization;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Drawing;
using System.Linq;
using System.Reflection.Emit;

namespace AutoEQ.Core;

public class PEQ
{
    public double fs { get; }
    public double[] FrequencyArray;
    public IList<double> f => FrequencyArray;
    private double[]? TargetArray;
    public IList<double>? target => TargetArray;
    public List<PEQFilter> filters { get; } = new();
    public OptimizationHistory history { get; private set; } = new();

    public double min_f { get; set; } = Constants.DEFAULT_PEQ_OPTIMIZER_MIN_F;
    public double max_f { get; set; } = Constants.DEFAULT_PEQ_OPTIMIZER_MAX_F;

    public TimeSpan? max_time { get; set; } = Constants.DEFAULT_PEQ_OPTIMIZER_MAX_TIME;
    public double? target_loss { get; set; } = Constants.DEFAULT_PEQ_OPTIMIZER_TARGET_LOSS;
    public double? min_change_rate { get; set; } = Constants.DEFAULT_PEQ_OPTIMIZER_MIN_CHANGE_RATE;
    public double? min_std { get; set; } = Constants.DEFAULT_PEQ_OPTIMIZER_MIN_STD;

    protected int Index_50hz => IndexClosestTo(50);
    protected int Index_10khz => IndexClosestTo(10000);
    protected int Index_20khz => IndexClosestTo(20000);

    protected int Index_MinFreq => IndexClosestTo(min_f);
    protected int Index_MaxFreq => IndexClosestTo(max_f);


    private int IndexClosestTo(double Freq) => f.Select(i => Math.Abs(i - Freq)).argMin();

    public PEQ(IEnumerable<double> f, double fs, IEnumerable<PEQFilter> filters) : this(f, fs) => this.filters.AddRange(filters);
    public PEQ(IEnumerable<double> f, double fs)
    {
        this.FrequencyArray = f.ToArray();
        this.fs = fs;
    }
    /*
    /// <summary>
    /// Initializes class instance with configuration dict and target
    /// </summary>
    /// <param name="config">
    /// Configuration dict with sampling rate "fs", filters and optionally filter defaults. Filters and
    /// filter defaults are dicts with keys fc, q, gain, min_fc, max_fc, min_q, max_q, min_gain, max_gain and
    /// type.The filter fc, q and gain are optimized if they are not present in the filter dicts, separately
    /// for each filter. "type" can be "LOW_SHEL$", "PEAKING" or "HIGH_SHEL$". "filter_defaults" sets the
    /// default values for filters to avoid repetition. Be wary of setting fc, q and gain in filter defaults
    /// as these will disable optimization for all filters and there is no way to enable optimization for a
    /// single filter after that.See `constants.py` for examples.
    /// </param>
    /// <param name="Frequency">Frequency array</param>
    /// <param name="fs">Sampling rate</param>
    /// <param name="Target">Equalizer frequency response target. Needed if optimization is to be performed.</param>
    /// <returns></returns>
    public static PEQ FromDict(Dictionary<string, object> config, IEnumerable<double> f, double fs, IEnumerable<double>? target = null)
    {

        if (target != null && f.Count != target.Count)
            throw new ArgumentException("f and target must be the same length");
        //optimizer_kwargs = config["optimizer"] if "optimizer" in config else {}
        PEQ peq = new(f, fs)
        {
            target = target.ToList()
        };
        //filter_classes = {"LOW_SHEL$": LowShelf, "PEAKING": Peaking, "HIGH_SHELF": HighShelf}
        ImmutableArray<string> keys = ImmutableArray.Create("fc", "q", "gain", "min_fc", "max_fc", "min_q", "max_q", "min_gain", "max_gain", "type");
        
        for filt in config["filters"]
            if("filter_defaults" in config)
                for key in keys:
                    if(key not in filt and key in config["filter_defaults"])
                        filt[key] = config["filter_defaults"][key]
            peq.add_filter(filter_classes[filt["type"]](
                peq.f, peq.fs,
                **{key: filt[key] for key in keys if key in filt and key != "type"},
                optimize_fc="fc" not in filt, optimize_q="q" not in filt, optimize_gain="gain" not in filt
            ))
        return peq;
    }
    */
    public void add_filter(PEQFilter filt)
    {
        if (filt.fs != fs)
            throw new ArgumentException($"Filter sampling rate ({filt.fs}) must match equalizer sampling rate ({fs})");
        if (filt.f.Zip(f).Any(p => p.First != p.Second))
            throw new ArgumentException("Filter frequency array (f) must match equalizer frequency array");
        filters.Add(filt);
    }
    public void sort_filters() => filters.Sort(PeqComparer.Instance);

    /// <summary>
    /// Calculates cascade frequency response
    /// </summary>
    public List<double> fr
    {
        get
        {
            List<double> fr = new(filters.First().fr);
            foreach (PEQFilter? f in filters.Skip(1))
            {
                for (int i = 0; i < fr.Count; i++)
                    fr[i] += f.fr[i];
            }
            return fr;
        }
    }

    /// <summary>
    /// Calculates maximum gain of frequency response
    /// </summary>
    public double max_gain => fr.Max();

    /// <summary>
    /// Formats filters as a Markdown table string"
    /// </summary>
    /// <returns></returns>
    public string markdown_table()
    {
        var table_data =
            filters.Select((f, i) =>
                new
                {
                    ID = i + 1,
                    Name = f.GetType().Name,
                    CenterFrequency = $"{f.fc:.0f}",
                    Q = $"{f.q:.2f}",
                    Gain = $"{f.gain:.1f}"
                });
        return table_data.ToMarkdownTable();
        /*
        return tabulate(
            table_data,
            headers =["#", "Type", "Fc (Hz)", "Q", "Gain (dB)"],
            tablefmt = "github"
        )*/
    }

    /// <summary>
    /// Extracts fc, q and gain from optimizer @params and updates filters
    /// </summary>
    /// <param name="@params">Parameter list/array passed by the optimizer. The values correspond to the initialized @params</param>
    public void _parse_optimizer_params(dynamic @params)
    {
        int i = 0;

        foreach (PEQFilter? filt in filters)
        {
            if (filt.@params_fc.HasValue)
            {
                filt.fc = Math.Pow(10, @params[i]);
                i += 1;
            }

            if (filt.@params_q.HasValue)
            {
                filt.q = @params[i];
                i += 1;
            }

            if (filt.@params_gain .HasValue)
            {
                filt.gain = @params[i];
                i += 1;
            }
        }
    }
    /// <summary>
    /// Calculates optimizer loss value
    /// </summary>
    /// <param name="@params"></param>
    /// <param name="parse"></param>
    /// <returns></returns>
    public double _optimizer_loss(dynamic @params, bool parse = true)
    {
        // Update filters with latest iteration @params
        if (parse)
            _parse_optimizer_params(@params);

        // Above 10 kHz only the total energy matters so we'll take the average
        var fr = this.fr.ToList();
        var target = this.target.ToList();
        target.SetRange(Index_10khz..(target.Count), target.EnumerateFrom(Index_10khz).Average());
        //target[self._10k_ix:] = np.mean(target[_10k_ix:])
        //fr[self._10k_ix:] = np.mean(self.fr[self._10k_ix:])
        fr.SetRange(Index_10khz..(fr.Count), fr.EnumerateFrom(Index_10khz).Average());
        //#target[:self._ix50] = np.mean(target[:self._ix50])  // TODO: Is this good?
        //#fr[:self._ix50] = np.mean(fr[:self._ix50])

        // Mean squared error as loss, between minimum and maximum frequencies
        double loss_val = target.Zip(fr, (t, r) => Math.Pow(t - r, 2)).Average();
        //double loss_val = (np.square(target[self._min_f_ix:self._max_f_ix] - fr[self._min_f_ix:self._max_f_ix])).Average;

        // Sum penalties from all filters to MSE
        loss_val += filters.Sum(f => f.sharpness_penalty);
        //#loss_val += filt.band_penalty  // TODO

        return Math.Sqrt(loss_val);
    }
    /// <summary>
    /// Creates a list of initial parameter values for the optimizer The list is fc, q and gain from each filter.Non-optimizable parameters are skipped.
    /// </summary>
    /// <param name=""></param>
    /// <returns></returns>
    private static readonly IList<(string Name, bool, bool)> order = ImmutableArray.Create(
            (typeof(Peaking).Name, true, true),  // Peaking
            (typeof(LowShelf).Name, true, true),  // Low shelfs
            (typeof(HighShelf).Name, true, true),  // High shelfs
            (typeof(Peaking).Name, true, false),  // Peaking with fixed q
            (typeof(LowShelf).Name, true, false),  // Low shelfs with fixed q
            (typeof(HighShelf).Name, true, false),  // High shelfs with fixed q
            (typeof(Peaking).Name, false, true),  // Peaking with fixed fc
            (typeof(LowShelf).Name, false, true),  // Low shelfs with fixed fc
            (typeof(HighShelf).Name, false, true),  // High shelfs with fixed fc
            (typeof(Peaking).Name, false, false),  // Peaking with fixed fc and q
            (typeof(LowShelf).Name, false, false),  // Low shelfs with fixed fc and q
            (typeof(HighShelf).Name, false, false)   // High shelfs with fixed fc and q
        );

    private IList<(double? CenterFrequency, double? Quality, double? Gain)> _init_optimizer_params()
    {
        double init_order(int filter_ix)
        {
            PEQFilter filt = filters[filter_ix];
            int ix = order.IndexOf((filt.GetType().Name, filt.@params_fc.HasValue, filt.@params_q.HasValue));
            double val = ix * 100;
            if (filt.@params_fc.HasValue)
                val += 1 / Math.Log2(filt.@params_fc.Value.max / filt.@params_fc.Value.min);
            return val;
        }

        // Initialize filter @params as list of empty lists, one per filter
        List<(double? CenterFrequency, double? Quality, double? Gain)>? filter_params = new(filters.Count);

        // Indexes to self.filters sorted by filter init order
        List<int> filter_argsort = Enumerable.Range(0, filters.Count).OrderByDescending(init_order).ToList();
        List<double> remaining_target = target.ToList();
        foreach (int ix in filter_argsort)  // Iterate sorted filter indexes
        {
            PEQFilter? filt = filters[ix];  // Get filter
            filter_params[ix] = filt.init(remaining_target);  // Init filter and place @params to list of lists
            remaining_target.SubZip(filt.fr);  // Adjust target
        }
//        filter_params = np.concatenate(filter_params).flatten()  // Flatten @params list
        return filter_params;
    }

    /// <summary>
    /// Creates optimizer bounds - For each optimizable fc, q and gain a (min, max) tuple is added
    /// </summary>
    /// <param name=""></param>
    /// <returns></returns>
    public IEnumerable<((double Min, double Max)? CenterFrequency, (double Min, double Max)? Quality, (double Min, double Max)? Gain)> _init_optimizer_bounds()
    {
        foreach(var filt in filters)
        {
            (double Min, double Max)? CenterFrequency = null;
            (double Min, double Max)? Quality = null;
            (double Min, double Max)? Gain = null;
            if (filt.@params_fc.HasValue)
                CenterFrequency = (Math.Log10(filt.@params_fc.Value.min), Math.Log10(filt.@params_fc.Value.max));
            if(filt.@params_q.HasValue)
                Quality = (filt.@params_q.Value.min, filt.@params_q.Value.max);
            if(filt.@params_gain.HasValue)
                Gain = (filt.@params_gain.Value.min, filt.@params_gain.Value.max);
            yield return (CenterFrequency, Quality, Gain);
        }
    }

    /// <summary>
    /// Optimization callback function
    /// </summary>
    /// <param name=""></param>
    private void _callback(dynamic @params)
    {
        TimeSpan t = history.SW.Elapsed;
        double loss = _optimizer_loss(@params, parse: false);
        //self.history.time.append(t)

        // Standard deviation of the last N loss values
        double std = history.LastStd(1);
        // Standard deviation of the last N/2 loss values
        double std_np2 = history.LastStd(0.5);

        double moving_avg_loss = history.Points.Count >= OptimizationHistory.n ? history.LastN().Select(p => p.loss).Average() : 0.0;

        double change_rate;
        if (history.Points.Count > 1)
        {
            var d_loss = loss - history.LastN(2).First().moving_avg_loss;
            var d_time = t - history.LastN(2).First().TimeTaken;
            change_rate = d_loss / d_time.TotalMilliseconds;
        }
        else
        {
            change_rate = 0.0;
        }

        OptimizationHistory.OptmizationPoint ThisPoint = new()
        {
            //    TimeTaken = t???,
            std = std,
            loss = loss,
            @params = @params,
            moving_avg_loss = moving_avg_loss,
            change_rate = change_rate
        };
        if (max_time.HasValue && history.SW.Elapsed >= max_time.Value)
            throw new OptimizationFinished("Maximum time reached");
        if(target_loss != null && loss <= target_loss)
            throw new OptimizationFinished("Target loss reached");
        if (min_change_rate.HasValue && history.Points.Count > OptimizationHistory.n && -change_rate < min_change_rate.Value)
            throw new OptimizationFinished("Change too small");
        if (min_std.HasValue && (
                // STD from last N loss values must be below min STD OR...
                (history.Points.Count > OptimizationHistory.n && std < min_std)
                // ...STD from the last N/2 loss values must be below half of the min STD
                || (history.Points.Count > Math.Floor(OptimizationHistory.n / 2.0) && std_np2 < (min_std / 2))))
            throw new OptimizationFinished("STD too small");
    }
    private NelderMeadSimplex? _Optimizer;
    private NelderMeadSimplex Optimizer => _Optimizer ??= new(.0005, 10000);
    /// <summary>
    /// Optimizes filter parameters
    /// </summary>
    /// <returns></returns>
    public dynamic optimize()
    {
        history = new();
        try
        {

            Optimizer.FindMinimum()
            Vector Guess = new DenseVector(new double[] { 1, 8 });
            var objective = ObjectiveFunction.Value(FuncWrap);
            MinimizationResult res = Opt.FindMinimum(objective, Guess);
            Console.WriteLine(res.ReasonForExit);
            Console.WriteLine(res.FunctionInfoAtMinimum.ToString());
            Console.WriteLine(res.MinimizingPoint.ToString());
            Console.WriteLine(FuncWrap(res.MinimizingPoint));            minimizer.


            fmin_slsqp(  // Tested all of the scipy minimize methods, this is the best
                _optimizer_loss,
                _init_optimizer_params(),
                bounds: self._init_optimizer_bounds(),
                callback = self._callback,
                iprint = 0);
        }
        catch (OptimizationFinished err)
        {

            // Restore best @params
            _parse_optimizer_params(history.@params[np.argmin(history.loss)]);
            Console.WriteLine(err);
        }
    }
}
