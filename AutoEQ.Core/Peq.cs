using AutoEQ.Core.Filters;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

namespace AutoEQ.Core;

public class PEQ
{
    public double fs { get; }
    public List<double> f { get; }
    public List<double>? target { get; set; }
    public List<PEQFilter> filters { get; } = new();
    public List<dynamic> history { get; } = new();

    public double min_f { get; set; } = Constants.DEFAULT_PEQ_OPTIMIZER_MIN_F;
    public double max_f { get; set; } = Constants.DEFAULT_PEQ_OPTIMIZER_MAX_F;

    public TimeSpan max_time { get; set; } = Constants.DEFAULT_PEQ_OPTIMIZER_MAX_TIME;
    public double target_loss { get; set; } = Constants.DEFAULT_PEQ_OPTIMIZER_TARGET_LOSS;
    public double min_change_rate { get; set; } = Constants.DEFAULT_PEQ_OPTIMIZER_MIN_CHANGE_RATE;
    public double min_std { get; set; } = Constants.DEFAULT_PEQ_OPTIMIZER_MIN_STD;


    public PEQ(IEnumerable<double> f, double fs, IEnumerable<PEQFilter> filters) : this(f, fs) => this.filters.AddRange(filters);
    public PEQ(IEnumerable<double> f, double fs)
    {
        this.f = f.ToList();
        this.fs = fs
        self._min_f_ix = np.argmin(np.abs(self.f - self._min_f))
        self._max_f_ix = np.argmin(np.abs(self.f - self._max_f))
        self._ix50 = np.argmin(np.abs(self.f - 50))
        self._10k_ix = np.argmin(np.abs(self.f - 10000))
        self._20k_ix = np.argmin(np.abs(self.f - 20000))
    }
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
    /// Extracts fc, q and gain from optimizer params and updates filters
    /// </summary>
    /// <param name="params">Parameter list/array passed by the optimizer. The values correspond to the initialized params</param>
    public void _parse_optimizer_params(dynamic @params)
    {
        int i = 0;

        foreach (PEQFilter? filt in filters)
        {
            if (filt.optimize_fc)
            {
                filt.fc = Math.Pow(10, @params[i]);
                i += 1;
            }

            if (filt.optimize_q)
            {
                filt.q = @params[i];
                i += 1;
            }

            if (filt.optimize_gain)
            {
                filt.gain = @params[i];
                i += 1;
            }
        }
    }
    def _optimizer_loss(self, @params, parse=true):
        """Calculates optimizer loss value"""
        # Update filters with latest iteration params
        if(parse)
            self._parse_optimizer_params(params)

        # Above 10 kHz only the total energy matters so we'll take the average
        fr = self.fr.copy()
        target = self.target.copy()
        target[self._10k_ix:] = np.mean(target[self._10k_ix:])
        fr[self._10k_ix:] = np.mean(self.fr[self._10k_ix:])
        #target[:self._ix50] = np.mean(target[:self._ix50])  # TODO: Is this good?
        #fr[:self._ix50] = np.mean(fr[:self._ix50])

        # Mean squared error as loss, between minimum and maximum frequencies
        loss_val = np.mean(np.square(target[self._min_f_ix:self._max_f_ix] - fr[self._min_f_ix:self._max_f_ix]))

        # Sum penalties from all filters to MSE
        for filt in self.filters:
            loss_val += filt.sharpness_penalty
            #loss_val += filt.band_penalty  # TODO

        return np.sqrt(loss_val)

    def _init_optimizer_params(self):
        """Creates a list of initial parameter values for the optimizer

        The list is fc, q and gain from each filter.Non-optimizable parameters are skipped.
        """
        var order = ImmutableArray.Create(
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
            (typeof(HighShelf).Name, false, false),  // High shelfs with fixed fc and q
        );

        def init_order(filter_ix):
            filt = self.filters[filter_ix]
            ix = order.index([filt.__class__typeof().Name, filt.optimize_fc, filt.optimize_q])
            val = ix * 100
            if(filt.optimize_fc)
                val += 1 / np.log2(filt.max_fc / filt.min_fc)
            return val

        # Initialize filter params as list of empty lists, one per filter
        filter_params = [[]] * len(self.filters)
        # Indexes to self.filters sorted by filter init order
        filter_argsort = sorted(list(range(len(self.filters))), key=init_order, reverse=true)
        remaining_target = self.target.copy()
        for ix in filter_argsort:  # Iterate sorted filter indexes
            filt = self.filters[ix]  # Get filter
            filter_params[ix] = filt.init(remaining_target)  # Init filter and place params to list of lists
            remaining_target -= filt.fr  # Adjust target
        filter_params = np.concatenate(filter_params).flatten()  # Flatten params list
        return filter_params

    def _init_optimizer_bounds(self):
        """Creates optimizer bounds

        For each optimizable fc, q and gain a (min, max) tuple is added
        """
        bounds = []
        for filt in self.filters:
            if(filt.optimize_fc)
                bounds.append((np.log10(filt.min_fc), np.log10(filt.max_fc)))
            if(filt.optimize_q)
                bounds.append((filt.min_q, filt.max_q))
            if(filt.optimize_gain)
                bounds.append((filt.min_gain, filt.max_gain))
        return bounds

    def _callback(self, params):
        """Optimization callback function"""
        n = 8
        t = time() - self.history.start_time
        loss = self._optimizer_loss(params, parse=false)

        self.history.time.append(t)
        self.history.loss.append(loss)

        # Standard deviation of the last N loss values
        std = np.std(np.array(self.history.loss[-n:]))
        # Standard deviation of the last N/2 loss values
        std_np2 = np.std(np.array(self.history.loss[-n//2:]))
        self.history.std.append(std)

        moving_avg_loss = np.mean(np.array(self.history.loss[-n:])) if len(self.history.loss) >= n else 0.0
        self.history.moving_avg_loss.append(moving_avg_loss)
        if(len(self.history.moving_avg_loss) > 1)
            d_loss = loss - self.history.moving_avg_loss[-2]
            d_time = t - self.history.time[-2]
            change_rate = d_loss / d_time if len(self.history.moving_avg_loss) > n else 0.0
        else:
            change_rate = 0.0
        self.history.change_rate.append(change_rate)
        self.history.params.append(params)
        if(self._max_time is not None and t >= self._max_time)
            raise OptimizationFinished("Maximum time reached")
        if(self._target_loss is not None and loss <= self._target_loss)
            raise OptimizationFinished("Target loss reached")
        if (
                self._min_change_rate is not None
                and len(self.history.moving_avg_loss) > n
                and -change_rate < self._min_change_rate
        ):
            raise OptimizationFinished("Change too small")
        if self._min_std is not None and (
                # STD from last N loss values must be below min STD OR...
                (len(self.history.std) > n and std < self._min_std)
                # ...STD from the last N/2 loss values must be below half of the min STD
                or (len(self.history.std) > n // 2 and std_np2 < self._min_std / 2)
        ):
            raise OptimizationFinished("STD too small")

    def optimize(self):
        """Optimizes filter parameters"""
        self.history = OptimizationHistory()
        try:
            fmin_slsqp(  # Tested all of the scipy minimize methods, this is the best
                self._optimizer_loss,
                self._init_optimizer_params(),
                bounds=self._init_optimizer_bounds(),
                callback=self._callback,
                iprint=0)
        except OptimizationFinished as err:
            # Restore best params
            self._parse_optimizer_params(self.history.params[np.argmin(self.history.loss)])
            #print(err)

    def plot(self, fig=None, ax=None):
        if(fig is None)
            fig, ax = plt.subplots()
            fig.set_size_inches(12, 8)
            fig.set_facecolor("white")
            ax.set_facecolor("white")
            ax.set_xlabel("Frequency (Hz)")
            ax.semilogx()
            ax.set_xlim([20, 20e3])
            ax.set_ylabel("Amplitude (dBr)")
            ax.grid(true, which="major")
            ax.grid(true, which="minor")
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
            ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        if(self.target is not None)
            ax.plot(self.f, self.target, color="black", linestyle="--", linewidth=1, label="Target")
        for i, filt in enumerate(self.filters):
            ax.fill_between(filt.f, np.zeros(filt.fr.shape), filt.fr, alpha=0.3, color=$"C{i}")
            ax.plot(filt.f, filt.fr, color=$"C{i}", linewidth=1)
        ax.plot(self.f, self.fr, color="black", linewidth=1, label="FR")
        ax.legend()
        return fig, ax


}