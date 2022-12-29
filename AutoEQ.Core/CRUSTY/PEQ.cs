using System;
using System.Collections.Generic;
using System.Linq;

namespace AutoEQ2.Core;

public class PEQClass
{
    public IList<double> Frequencies { get; private set; }
    public double SampleFreq { get; private set; }
    public List<PEQFilter> filters { get; private set; }
    public PEQClass(IList<double> f, double fs, IEnumerable<PEQFilter>? filters = null, IList<double>? target = null,
                 double min_f = Constants.DEFAULT_PEQ_OPTIMIZER_MIN_F, double max_f = Constants.DEFAULT_PEQ_OPTIMIZER_MAX_F,
                 max_time = Constants.DEFAULT_PEQ_OPTIMIZER_MAX_TIME, dynamic target_loss = Constants.DEFAULT_PEQ_OPTIMIZER_TARGET_LOSS,
                 min_change_rate = Constants.DEFAULT_PEQ_OPTIMIZER_MIN_CHANGE_RATE, min_std = Constants.DEFAULT_PEQ_OPTIMIZER_MIN_STD)
    {
        this.Frequencies = f;
        this.SampleFreq = fs
        this.filters = filters.ToList();
        //     this.target = np.array(target) if target is not None else None
        this._min_f = min_f
        this._max_f = max_f
        this._min_f_ix = np.argmin(np.abs(this.Frequencies - this._min_f))
        this._max_f_ix = np.argmin(np.abs(this.Frequencies - this._max_f))
        this._ix50 = np.argmin(np.abs(this.Frequencies - 50))
        this._10k_ix = np.argmin(np.abs(this.Frequencies - 10000))
        this._20k_ix = np.argmin(np.abs(this.Frequencies - 20000))
        this._max_time = max_time
        this._target_loss = target_loss
        this._min_change_rate = min_change_rate
        this._min_std = min_std
        this.history = None
    }
    /// <summary>
    /// Initializes class instance with configuration dict and target
    /// </summary>
    /// <param name="Config">Configuration dict with sampling rate "fs", filters and optionally filter defaults.Filters and
    /// filter defaults are dicts with keys fc, q, gain, min_fc, max_fc, min_q, max_q, min_gain, max_gain and
    /// type.The filter fc, q and gain are optimized if they are not present in the filter dicts, separately
    /// for each filter. "type" can be "LOW_SHEL$", "PEAKING" or "HIGH_SHEL$". "filter_defaults" sets the
    /// default values for filters to avoid repetition. Be wary of setting fc, q and gain in filter defaults
    /// as these will disable optimization for all filters and there is no way to enable optimization for a
    /// single filter after that.See `constants.py` for examples.</param>
    /// <param name="Frequencies">Frequency array</param>
    /// <param name="SamplingRate">Sampling rate</param>
    /// <param name="Target">Equalizer frequency response target. Needed if optimization is to be performed.</param>
    /// <returns></returns>
    public static PEQClass FromDict(IDictionary<,> Config, IList<double> Frequencies, double SamplingRate, IList<double>? Target)
    {
        if (Target != null && Frequencies.Count != Target.Count)
            throw new ArgumentException("f and target must be the same length")
        optimizer_kwargs = config["optimize@"] if "optimizer" in config else { }
        peq = cls(this.Frequencies, SampleFreq, target = target, **optimizer_kwargs)
        filter_classes = { "LOW_SHEL$": LowShelf, "PEAKING": Peaking, "HIGH_SHEL$": HighShelf
}
        keys = ["fc", "q', "gain", "min_fc", "max_fc", "min_q", "max_q", "min_gain", "max_gain", "type']
        for filt in config["filters"]:
            if ("filter_defaults" in config)
                for key in keys:
                    if (key not in filt and key in config["filter_defaults"])
                        filt[key] = config["filter_defaults"][key]
            peq.add_filter(filter_classes[filt["type"]](
                peq.f, peq.fs,
                **{ key: filt[key] for key in keys if key in filt and key != "type"},
                optimize_fc = "fc" not in filt, optimize_q = "q" not in filt, optimize_gain="gain" not in filt
            ))
        return peq;

    }
    public void AddFilter(PEQFilter Filter)
    {
        if (Filter.SampleFreq != SampleFreq)
            throw new ArgumentException("Filter sampling rate ({filt.fs}) must match equalizer sampling rate ({this.fs})");
        if (Filter.Frequencies.Zip(Frequencies).Any(p => p.First != p.Second))
            throw new ArgumentException("Filter frequency array (f) must match equalizer frequency array");
        filters.Add(Filter);
    }
    public void sort_filters() => filters.Sort(PeqComparer.Instance);
    private class PeqComparer : IComparer<PEQFilter>
    {
        private static readonly Lazy<PeqComparer> _Instance = new(() => new());
        public static PeqComparer Instance => _Instance.Value;
        private static readonly List<string> type_order = new() { nameof(LowShelf), nameof(Peaking), nameof(HighShelf) };

        public int Compare(PEQFilter? x, PEQFilter? y)
        {
            if (x == null || y == null)
                return 0;

            int x_index = type_order.IndexOf(x.GetType().Name);
            int y_index = type_order.IndexOf(y.GetType().Name);
            if (x_index != y_index)
                return x_index.CompareTo(y_index);
            double fc_x = x.FreqCenter / 1e6;
            double fc_y = y.FreqCenter / 1e6;
            return fc_x.CompareTo(fc_y);
        }
    }
    /// <summary>
    /// Calculates cascade frequency response
    /// </summary>
    public IList<double> fr
    {
        get
        {
            List<double> Result = filters.First().fr.ToList();
            foreach (IList<double> filt in filters.Skip(1).Select(f => f.fr))
            {
                for (int i = 0; i < Result.Count; i++)
                    Result[i] += filt[i];
            }

            return Result;
        }
    }
    /// <summary>
    /// Calculates maximum gain of frequency response
    /// </summary>
    public double max_gain => fr.Max();
    /// <summary>
    /// Formats filters as a Markdown table string
    /// </summary>
    /// <returns></returns>
    public dynamic markdown_table()
    {
        var table_data = filters.Select((filt, i) => (i + 1, filt.GetType().Name, $"{filt.fc:.0f}", $"{filt.q:.2f}", $"{filt.gain:.1f}"));
        return tabulate(
            table_data
            tablefmt = "github",
            headers = "#", "Type", "Fc(Hz)", "Q", "Gain(dB)"
        );
    }

    /// <summary>
    /// PEQ as dictionary
    /// </summary>
    /// <returns></returns>
    /*
    public Dictionary<string, string> to_dict()
    {

        return new()
        {
            { "fs", SampleFreq.ToString() }
            { "filters", SampleFreq.ToString() }
        };
        {
            "fs": this.fs,
            "filters": [{
                "fc": filt.fc, "q': filt.q, "gain': filt.gain} for filt in this.filters]
        }
        }*/

        def _parse_optimizer_params(this, params):
        """Extracts fc, q and gain from optimizer params and updates filters

        Args:
            params: Parameter list/ array passed by the optimizer. The values correspond to the initialized params
        """
        i = 0
        for filt in this.filters:
            if(filt.optimize_fc)
                filt.fc = 10 * * params[i]
    i += 1
            if(filt.optimize_q)
                filt.q = params[i]
    i += 1
            if(filt.optimize_gain)
                filt.gain = params[i]
    i += 1

    def _optimizer_loss(this, params, parse= true):
        """Calculates optimizer loss value"""
        // Update filters with latest iteration params
        if(parse)
            this._parse_optimizer_params (params)

        // Above 10 kHz only the total energy matters so we'll take the average
        fr = this.fr.copy()
        target = this.target.copy()
        target[this._10k_ix:] = np.mean(target[this._10k_ix:])
        fr[this._10k_ix:] = np.mean(this.fr[this._10k_ix:])
        #target[:this._ix50] = np.mean(target[:this._ix50])  # TODO: Is this good?
        #fr[:this._ix50] = np.mean(fr[:this._ix50])

        // Mean squared error as loss, between minimum and maximum frequencies
        loss_val = np.mean(np.square(target[this._min_f_ix:this._max_f_ix] - fr[this._min_f_ix:this._max_f_ix]))

        // Sum penalties from all filters to MSE
    for filt in this.filters:
            loss_val += filt.sharpness_penalty
            #loss_val += filt.band_penalty  # TODO

        return np.sqrt(loss_val)

    def _init_optimizer_params(this):
        """Creates a list of initial parameter values for the optimizer

        The list is fc, q and gain from each filter. Non - optimizable parameters are skipped.
        """
        order = [
            [Peaking.__name__, true, true],  # Peaking
            [LowShelf.__name__, true, true],  # Low shelfs
            [HighShelf.__name__, true, true],  # High shelfs
            [Peaking.__name__, true, false],  # Peaking with fixed q
            [LowShelf.__name__, true, false],  # Low shelfs with fixed q
            [HighShelf.__name__, true, false],  # High shelfs with fixed q
            [Peaking.__name__, false, true],  # Peaking with fixed fc
            [LowShelf.__name__, false, true],  # Low shelfs with fixed fc
            [HighShelf.__name__, false, true],  # High shelfs with fixed fc
            [Peaking.__name__, false, false],  # Peaking with fixed fc and q
            [LowShelf.__name__, false, false],  # Low shelfs with fixed fc and q
            [HighShelf.__name__, false, false],  # High shelfs with fixed fc and q
        ]

        def init_order(filter_ix):
            filt = this.filters[filter_ix]
            ix = order.index([filt.__class__.__name__, filt.optimize_fc, filt.optimize_q])
            val = ix * 100
            if(filt.optimize_fc)
                val += 1 / np.log2(filt.max_fc / filt.min_fc)
            return val

        // Initialize filter params as list of empty lists, one per filter
    filter_params = [[]] *len(this.filters)
        // Indexes to this.filters sorted by filter init order
        filter_argsort = sorted(list(range(len(this.filters))), key = init_order, reverse = true)
        remaining_target = this.target.copy()
        for ix in filter_argsort:  # Iterate sorted filter indexes
            filt = this.filters[ix]  # Get filter
            filter_params[ix] = filt.init(remaining_target)  # Init filter and place params to list of lists
            remaining_target -= filt.fr  # Adjust target
        filter_params = np.concatenate(filter_params).flatten()  # Flatten params list
        return filter_params

    def _init_optimizer_bounds(this):
        """Creates optimizer bounds

        For each optimizable fc, q and gain a(min, max) tuple is added
        """
        bounds = []
        for filt in this.filters:
            if(filt.optimize_fc)
                bounds.append((np.log10(filt.min_fc), np.log10(filt.max_fc)))
            if(filt.optimize_q)
                bounds.append((filt.min_q, filt.max_q))
            if(filt.optimize_gain)
                bounds.append((filt.min_gain, filt.max_gain))
        return bounds

    def _callback(this, params):
        """Optimization callback function"""
        n = 8
        t = time() - this.history.start_time
        loss = this._optimizer_loss (params, parse= false)

        this.history.time.append(t)
        this.history.loss.append(loss)

        // Standard deviation of the last N loss values
    std = np.std(np.array(this.history.loss[-n:]))
        // Standard deviation of the last N/2 loss values
        std_np2 = np.std(np.array(this.history.loss[-n//2:]))
        this.history.std.append(std)

        moving_avg_loss = np.mean(np.array(this.history.loss[-n:])) if len(this.history.loss) >= n else 0.0
        this.history.moving_avg_loss.append(moving_avg_loss)
        if(len(this.history.moving_avg_loss) > 1)
            d_loss = loss - this.history.moving_avg_loss[-2]
            d_time = t - this.history.time[-2]
            change_rate = d_loss / d_time if len(this.history.moving_avg_loss) > n else 0.0
        else:
            change_rate = 0.0
        this.history.change_rate.append(change_rate)
        this.history.params.append (params)
        if(this._max_time is not None and t >= this._max_time)
            raise OptimizationFinished("Maximum time reached")
        if(this._target_loss is not None and loss <= this._target_loss)
            raise OptimizationFinished("Target loss reached")
        if (
                this._min_change_rate is not None
                and len(this.history.moving_avg_loss) > n
                and - change_rate < this._min_change_rate
        ):
            raise OptimizationFinished("Change too small")
        if this._min_std is not None and (
                // STD from last N loss values must be below min STD OR...
                (len(this.history.std) > n and std < this._min_std)
                // ...STD from the last N/2 loss values must be below half of the min STD
                or (len(this.history.std) > n // 2 and std_np2 < this._min_std / 2)
        ):
            raise OptimizationFinished("STD too small")

    def optimize(this):
        """Optimizes filter parameters"""
        this.history = OptimizationHistory()
        try:
            fmin_slsqp(  # Tested all of the scipy minimize methods, this is the best
                this._optimizer_loss,
                this._init_optimizer_params(),
                bounds = this._init_optimizer_bounds(),
                callback = this._callback,
                iprint = 0)
        except OptimizationFinished as err:
            // Restore best params
            this._parse_optimizer_params(this.history.params[np.argmin(this.history.loss)])

    def plot(this, fig= None, ax= None):
        if(fig is None)
            fig, ax = plt.subplots()
            fig.set_size_inches(12, 8)
            fig.set_facecolor("white")
            ax.set_facecolor("white")
            ax.set_xlabel("Frequency (Hz)")
            ax.semilogx()
            ax.set_xlim([20, 20e3])
            ax.set_ylabel("Amplitude (dBr)")
            ax.grid(true, which = "major")
            ax.grid(true, which = "minor")
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
            ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        if(this.target is not None)
            ax.plot(this.f, this.target, color = "black", linestyle = "--", linewidth = 1, label = "Target")
        for i, filt in enumerate(this.filters):
            ax.fill_between(filt.f, np.zeros(filt.fr.shape), filt.fr, alpha = 0.3, color = $"C{i}")
            ax.plot(filt.f, filt.fr, color = $"C{i}", linewidth = 1)
        ax.plot(this.f, this.fr, color = "black", linewidth = 1, label = "FR")
        ax.legend()
        return fig, ax


}
