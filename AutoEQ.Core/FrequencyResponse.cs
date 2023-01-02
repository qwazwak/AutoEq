using AutoEQ.Core.Filters;
using AutoEQ.Core.Models;
using AutoEQ.Helper;
using CsvHelper;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace AutoEQ.Core;

public class PeqComparer : IComparer<PEQFilter>
{
    private static readonly Lazy<PeqComparer> _Instance = new(() => new());
    public static PeqComparer Instance => _Instance.Value;
    private static readonly ImmutableArray<Type> type_order = ImmutableArray.Create(typeof(LowShelf), typeof(Peaking), typeof(HighShelf));

    public int Compare(PEQFilter? x, PEQFilter? y)
    {
        if (x == null || y == null)
            return 0;

        int x_index = type_order.IndexOf(x.GetType());
        int y_index = type_order.IndexOf(y.GetType());
        if (x_index != y_index)
            return x_index.CompareTo(y_index);
        if (!x.fc.HasValue || !y.fc.HasValue)
            return 0;
        double fc_x = x.fc.Value / 1e6;
        double fc_y = y.fc.Value / 1e6;
        return fc_x.CompareTo(fc_y);
    }
}

public class FrequencyResponse
{
    public string name { get; }
   // public List<ResponsePoint> Response { get; } = new();
    public List<double> frequency { get; private set; }
    public List<double> raw { get; private set; }
    public List<double> error { get; private set; }
    public List<double> smoothed { get; private set; }
    public List<double> error_smoothed { get; private set; }
    public List<double> equalization { get; private set; }
    public List<double> parametric_eq { get; private set; }
    public List<double> fixed_band_eq { get; private set; }
    public IList<double> equalized_raw { get; private set; }//{ get => ; init => AddInitData(value, (i, v) => Response[i].Equalized_raw = v); }
    public List<double> equalized_smoothed { get; private set; }
    public List<double> target { get; private set; }

    public FrequencyResponse(string name, IEnumerable<double>? frequency,
        IEnumerable<double>? raw = null, IEnumerable<double>? error = null, IEnumerable<double>? smoothed = null, IEnumerable<double>? error_smoothed = null,
        IEnumerable<double>? equalization = null, IEnumerable<double>? parametric_eq = null, IEnumerable<double>? fixed_band_eq = null,
        IEnumerable<double>? equalized_raw = null, IEnumerable<double>? equalized_smoothed = null, IEnumerable<double>? target = null)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Name must not be a non-empty string.");
        this.name = name.Trim();

        this.frequency = _init_data(frequency);
        if (this.frequency.Count == 0)
            this.frequency = generate_frequencies();

        this.raw = _init_data(raw);
        this.smoothed = _init_data(smoothed);
        this.error = _init_data(error);
        this.error_smoothed = _init_data(error_smoothed);
        this.equalization = _init_data(equalization);
        this.parametric_eq = _init_data(parametric_eq);
        this.fixed_band_eq = _init_data(fixed_band_eq);
        this.equalized_raw = _init_data(equalized_raw);
        this.equalized_smoothed = _init_data(equalized_smoothed);
        this.target = _init_data(target);

//        Response.Sort();
        this._sort();

    }

    private void _sort()
    {
        var sorted_inds = frequency.GetSortingIndexs().ToList();
        frequency = frequency.SortByIndexes(sorted_inds);
        foreach (IGrouping<double, double>? item in frequency.GroupBy(i => i).Where(i => i.Count() > 1))
            throw new ArgumentException($"Duplicate values found at frequency {item.Key}. Remove duplicates manually.");

        if (raw.Count > 0)
            raw = raw.SortByIndexes(sorted_inds);
        if (error.Count > 0)
            error = error.SortByIndexes(sorted_inds);
        if (smoothed.Count > 0)
            smoothed = smoothed.SortByIndexes(sorted_inds);
        if (error_smoothed.Count > 0)
            error_smoothed = error_smoothed.SortByIndexes(sorted_inds);
        if (equalization.Count > 0)
            equalization = equalization.SortByIndexes(sorted_inds);
        if (parametric_eq.Count > 0)
            parametric_eq = parametric_eq.SortByIndexes(sorted_inds);
        if (fixed_band_eq.Count > 0)
            fixed_band_eq = fixed_band_eq.SortByIndexes(sorted_inds);
        if (equalized_raw.Count > 0)
            equalized_raw = equalized_raw.SortByIndexes(sorted_inds);
        if (equalized_smoothed.Count > 0)
            equalized_smoothed = equalized_smoothed.SortByIndexes(sorted_inds);
        if (target.Count > 0)
            target = target.SortByIndexes(sorted_inds);
    }

    private static void AddInitData(IList<double>? data, Action<int, double> SetValue)
    {
        if (data == null)
            return;
        for (int i = 0; i < data.Count; i++)
        {
            double? d = data[i];
            if (!d.HasValue)
                throw new ArgumentException();
            SetValue.Invoke(i, d.Value);
        }
    }

    /// <summary>
    /// Initializes data to a clean format. If None is passed and empty array is created. Non-numbers are removed.
    /// </summary>
    /// <param name="data"></param>
    /// <returns></returns>
    private static List<double> _init_data(IEnumerable<float>? data) => _init_data(data?.Cast<double>());
    private static List<double> _init_data(IEnumerable<int>? data) => _init_data(data?.Cast<double>());
    private static List<double> _init_data(IEnumerable<double>? data) => (data ?? Enumerable.Empty<double>()).Where(d => !double.IsNaN(d) && d != null).ToList();
    private static List<double> _init_data() => new();
    private static List<double> generate_frequencies(double f_min = Constants.DEFAULT_F_MIN, double f_max = Constants.DEFAULT_F_MAX, double f_step = Constants.DEFAULT_STEP)
        => EnumerateFrequencies(f_min, f_max, f_step).ToList();
    private static IEnumerable<double> EnumerateFrequencies(double f_min = Constants.DEFAULT_F_MIN, double f_max = Constants.DEFAULT_F_MAX, double f_step = Constants.DEFAULT_STEP)
    {
        double f = f_min;
        while (f <= f_max)
        {
            yield return f;
            f *= f_step;
        }
    }
    public IEnumerable<double> _sigmoid(double f_lower, double f_upper, double a_normal = 0.0, double a_treble = 1.0)
    {
        double f_center = Math.Sqrt(f_upper / f_lower) * f_lower;
        double half_range = Math.Log10(f_upper) - Math.Log10(f_center);
        f_center = Math.Log10(f_center);
        var a = this.frequency.Select(f => expit((Math.Log10(f) - f_center) / (half_range / 4)));
        return a.Select(i => (i * -(a_normal - a_treble)) + a_normal);

        static double expit(double x) => 1 / (1 + Math.Exp(-x));
    }

    /// <summary>
    /// Calculates Harman preference score for in-ear headphones.
    /// </summary>
    /// <returns>score: Preference score, std: Standard deviation of error, slope: Slope of linear regression of error, mean: Mean of absolute error</returns>
    public (double Score, double Std, double Slope, double Mean) harman_inear_preference_score()
    {
        FrequencyResponse fr = copy();
        fr.interpolate(Constants.HARMAN_INEAR_PREFENCE_FREQUENCIES);
        IEnumerable<bool> sl = fr.frequency.Select(f => f >= 20 && f <= 10000).ToList();
        IEnumerable<double> x = fr.frequency.SelectWhere(sl);
        IEnumerable<double> y = fr.error.SelectWhere(sl);

        //ddof=1 is required to get the exact same numbers as the Excel from Listen Inc gives
        double std = MathEx.StdDiv(y, ddof: 1);
        Regressor.LinearRegression(x.Select(x => Math.Log(x)), y, out double _, out double _, out double slope);
        // Mean of absolute of error centered by 500 Hz
        //double delta = fr.error[np.where(fr.frequency == 500.0)[0][0]];
        double delta = fr.error.SelectIndexes(fr.frequency.IndexWhere(f => f == 500)).First();
//        y = fr.error[Math.Logical_and(fr.frequency >= 40, fr.frequency <= 10000)] - delta;
        double mean = fr.error.SelectIndexes(fr.frequency.IndexWhere(f => f >= 40 && f <= 10000)).Select(i => i - delta).Select(Math.Abs).Average();
        // Final score
        double score = 100.0795 - (8.5 * std) - (6.796 * Math.Abs(slope)) - 3.475 * mean;

        return (score, std, slope, mean);
    }

    /// <summary>
    /// Reads data from CSV file and constructs class instance.
    /// </summary>
    /// <param name="file_path"></param>
    /// <returns></returns>
    public static async Task<FrequencyResponse> read_from_csvAsync(FileInfo file_path)
    {
        //name = ".".join(os.path.split(file_path)[1].split(".")[:-1])
        string name = file_path.Name;

        // Read file
        using FileStream fs = File.OpenRead(file_path.FullName);
        using StreamReader sr = new(fs);
        string? s = await sr.ReadLineAsync();

        // Regex for AutoEq style CSV
        string header_pattern = @"frequency(?:,(?:raw|smoothed|error|error_smoothed|equalization|parametric_eq|fixed_band_eq|equalized_raw|equalized_smoothed|target))+";
        string float_pattern = @"-?\d+(?:\.\d+)?";
        string data_2_pattern = @$"{float_pattern}[ ,;:\t]+{float_pattern}?";
        string data_n_pattern = @$"{float_pattern}(?:[ ,;:\t]+{float_pattern})+?";
        Regex autoeq_pattern = new(@$"^{header_pattern}(?:\n{data_n_pattern})+\n*$");

        if (!string.IsNullOrWhiteSpace(s) && autoeq_pattern.IsMatch(s))
        {
            List<double> frequency = new();
            List<double> raw = new();
            List<double> smoothed = new();
            List<double> error = new();
            List<double> error_smoothed = new();
            List<double> equalization = new();
            List<double> parametric_eq = new();
            List<double> fixed_band_eq = new();
            List<double> equalized_raw = new();
            List<double> equalized_smoothed = new();
            List<double> target = new();
            using (CsvReader csv = new(sr, CultureInfo.InvariantCulture))
            {
                await foreach (ResponsePoint r in csv.GetRecordsAsync<ResponsePoint>())
                {
                    frequency.Add(r.Frequency);

                    raw.AddIfNotNull(r.Raw);
                    smoothed.AddIfNotNull(r.Smoothed);
                    error.AddIfNotNull(r.Error);
                    error_smoothed.AddIfNotNull(r.Error_smoothed);
                    equalization.AddIfNotNull(r.Equalization);
                    parametric_eq.AddIfNotNull(r.Parametric_eq);
                    fixed_band_eq.AddIfNotNull(r.Fixed_band_eq);
                    equalized_raw.AddIfNotNull(r.Equalized_raw);
                    equalized_smoothed.AddIfNotNull(r.Equalized_smoothed);
                    target.AddIfNotNull(r.Target);
                }
            }
            // Known AutoEq CSV format
            return new(
                name: name,
                frequency: frequency,
                raw: raw,
                smoothed: smoothed,
                error: error,
                error_smoothed: error_smoothed,
                equalization: equalization,
                parametric_eq: parametric_eq,
                fixed_band_eq: fixed_band_eq,
                equalized_raw: equalized_raw,
                equalized_smoothed: equalized_smoothed,
                target: target
            );
        }
        else
        {
            throw new InvalidDataException();
            /*
            // Unknown format, try to guess
            lines = s.split("\n")
            frequency = []
            raw = []
            for line in lines:
                if (re.match(data_2_pattern, line))  // float separator float
                    floats = re.findall(float_pattern, line)
                    frequency.append(float(floats[0]))  // Assume first to be frequency
                    raw.append(float(floats[1]))  // Assume second to be raw
                // Discard all lines which don't match data pattern
            return cls(name = name, frequency = frequency, raw = raw);*/
        }
    }


    //warnings.filterwarnings("ignore", message="Values in x were outside bounds during a minimize step, clipping to bounds")

    public FrequencyResponse copy(string? name = null) => new(
        name: name ?? (this.name + "_copy"),
        frequency: _init_data(frequency),
        raw: _init_data(raw),
        error: _init_data(error),
        smoothed: _init_data(smoothed),
        error_smoothed: _init_data(error_smoothed),
        equalization: _init_data(equalization),
        parametric_eq: _init_data(parametric_eq),
        fixed_band_eq: _init_data(fixed_band_eq),
        equalized_raw: _init_data(equalized_raw),
        equalized_smoothed: _init_data(equalized_smoothed),
        target: _init_data(target)
    );
    public void reset(IDictionary<string, bool> Dict) => reset(Dict.Where(kvp => kvp.Value).Select(kvp => kvp.Key));
    public void reset(IEnumerable<string> ToReset) => reset(ToReset as ISet<string> ?? ToReset.ToHashSet());
    public void reset(ISet<string> ToReset) => reset(raw: ToReset.Contains("raw"),
            smoothed: ToReset.Contains("smoothed"),
            error: ToReset.Contains("error"),
            error_smoothed: ToReset.Contains("error_smoothed"),
            equalization: ToReset.Contains("equalization"),
            fixed_band_eq: ToReset.Contains("fixed_band_eq"),
            parametric_eq: ToReset.Contains("parametric_eq"),
            equalized_raw: ToReset.Contains("equalized_raw"),
            equalized_smoothed: ToReset.Contains("equalized_smoothed"),
            target: ToReset.Contains("target"));
    /// <summary>
    /// Resets data
    /// </summary>
    /// <param name="raw"></param>
    /// <param name="smoothed"></param>
    /// <param name="error"></param>
    /// <param name="error_smoothed"></param>
    /// <param name="equalization"></param>
    /// <param name="fixed_band_eq"></param>
    /// <param name="parametric_eq"></param>
    /// <param name="equalized_raw"></param>
    /// <param name="equalized_smoothed"></param>
    /// <param name="target"></param>
    /// <returns></returns>
    public void reset(
              bool raw = false,
              bool smoothed = true,
             bool error = true,
             bool error_smoothed = true,
            bool equalization = true,
            bool fixed_band_eq = true,
            bool parametric_eq = true,
            bool equalized_raw = true,
           bool equalized_smoothed = true,
           bool target = true)
    {
        if (raw)
            this.raw = _init_data();
        if (smoothed)
            this.smoothed = _init_data();
        if (error)
            this.error = _init_data();
        if (error_smoothed)
            this.error_smoothed = _init_data();
        if (equalization)
            this.equalization = _init_data();
        if (parametric_eq)
            this.parametric_eq = _init_data();
        if (fixed_band_eq)
            this.fixed_band_eq = _init_data();
        if (equalized_raw)
            this.equalized_raw = _init_data();
        if (equalized_smoothed)
            this.equalized_smoothed = _init_data();
        if (target)
            this.target = _init_data();
    }

    public IDictionary<string, object> to_dict()
    {
        Dictionary<string, object> d = new();
        if (frequency.Count > 0)
            d["frequency"] = frequency.ToList();

        //d["raw"] = [x if x is not None else "NaN" for x in raw];
        if (raw.Count > 0)
            d["raw"] = raw.Cast<double?>().Select(x => x?.ToString() ?? "NaN");
        if (error.Count > 0)
            d["error"] = error.Cast<double?>().Select(x => x?.ToString() ?? "NaN");
        if (smoothed.Count > 0)
            d["smoothed"] = smoothed.Cast<double?>().Select(x => x?.ToString() ?? "NaN");
        if (error_smoothed.Count > 0)
            d["error_smoothed"] = error_smoothed.Cast<double?>().Select(x => x?.ToString() ?? "NaN");
        if (equalization.Count > 0)
            d["equalization"] = equalization.Cast<double?>().Select(x => x?.ToString() ?? "NaN");
        if (parametric_eq.Count > 0)
            d["parametric_eq"] = parametric_eq.Cast<double?>().Select(x => x?.ToString() ?? "NaN");
        if (fixed_band_eq.Count > 0)
            d["fixed_band_eq"] = fixed_band_eq.Cast<double?>().Select(x => x?.ToString() ?? "NaN");
        if (equalized_raw.Count > 0)
            d["equalized_raw"] = equalized_raw.Cast<double?>().Select(x => x?.ToString() ?? "NaN");
        if (equalized_smoothed.Count > 0)
            d["equalized_smoothed"] = equalized_smoothed.Cast<double?>().Select(x => x?.ToString() ?? "NaN");
        if (target.Count > 0)
            d["target"] = target.Cast<double?>().Select(x => x?.ToString() ?? "NaN");
        return d;
    }
    /// <summary>
    /// Writes data to files as CSV.
    /// </summary>
    /// <param name=""></param>
    /// <returns></returns>
    public void write_to_csv(FileInfo file_path)
    {
        using FileStream fs = File.OpenWrite(file_path.FullName);
        using StreamWriter sw = new(fs);
        write_to_csv(sw);
     }   
    public void write_to_csv(TextWriter Output)
    {/*
        using CsvWriter csv = new(Output, CultureInfo.InvariantCulture, false);
        IDictionary<string, object>? Dict = to_dict();
        foreach (KeyValuePair<string, object> item in )
        {

        }
        csv.WriteHeader<>
            file_path = os.path.abspath(file_path)
            df = pd.DataFrame(self.to_dict())
            df.to_csv(file_path, header = true, index = false, float_format = "%.2f")
     */}

    public static IEnumerable<double> Arrange(double End, double Step = 1)
        => Arrange(0, End, Step);
    public static IEnumerable<double> Arrange(double Start, double End, double Step = 1)
    {
        double c = Start;
        while(c <= End)
        {
            yield return c;
            c += Step;
        }
    }

    /// <summary>
    /// Generates EqualizerAPO GraphicEQ string from equalization curve.
    /// </summary>
    /// <param name="normalize"></param>
    /// <param name="preamp"></param>
    /// <param name="f_step"></param>
    /// <returns></returns>
    public string eqapo_graphic_eq(bool normalize=true, double? preamp = Constants.DEFAULT_PREAMP, double f_step= Constants.DEFAULT_GRAPHIC_EQ_STEP)
    {
        FrequencyResponse fr = new(name: "hack", frequency: frequency, raw: equalization);
        double n = Math.Ceiling(Math.Log(20000 / 20) / Math.Log(f_step));
        IEnumerable<int> f = Arrange(n).Select(i => 20 * Math.Pow(f_step, i))
            .Cast<int>().Distinct().OrderByDescending(i => i);
        fr.interpolate(f: f.Cast<double>());
        if (normalize)
        {
            double RawMax = fr.raw.Max();
            fr.raw.ApplyEach(v => v - RawMax + Constants.PREAMP_HEADROOM);
        }
        if (preamp.HasValue)
            fr.raw.ApplyEach(v => v += preamp.Value);
        if (fr.raw[0] > 0.0)
        {
            // Prevent bass boost below lowest frequency
            fr.raw[0] = 0.0;
        }
        return  "GraphicEQ: " + string.Join("; ", fr.frequency.Zip(fr.raw, (f, a) => $"{f} {a:.1f}"));
    }

    /*
        public dynamic write_eqapo_graphic_eq(file_path, normalize=true, preamp=Constants.DEFAULT_PREAMP):
            """Writes equalization graph to a file as Equalizer APO config."""
            file_path = os.path.abspath(file_path)
            s = self.eqapo_graphic_eq(normalize=normalize, preamp=preamp)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(s)
            return s
    */
        public dynamic _optimize_peq_filters(configs, fs, max_time=None, double? preamp = Constants.DEFAULT_PREAMP)
    {
        return _optimize_peq_filters(new configs[] { configs }, fs, max_time, preamp);
    }
        public dynamic _optimize_peq_filters(IEnumerable<IDictionary<string, object>> configs, fs, TimeSpan? max_time=null, double? preamp=Constants.DEFAULT_PREAMP)
    {
        ICollection<PEQ> peqs = new List<PEQ>();
        FrequencyResponse fr = new(name: "optimizer", frequency: frequency, equalization: equalization);
        if (preamp.HasValue)
            fr.equalization.ForEach(v => v + preamp);
        fr.interpolate(f_step: Constants.DEFAULT_BIQUAD_OPTIMIZATION_F_STEP);
        DateTime start_time = DateTime.UtcNow;
        Stopwatch sw = Stopwatch.StartNew();
        foreach (IDictionary<string, object> config in configs)
        {
            if (config.ContainsKey("optimizer") && max_time.HasValue)
                config["optimizer"]["max_time"] = max_time;
            PEQ peq = PEQ.from_dict(config, fr.frequency, fs, target = fr.equalization);
            peq.optimize();
            fr.equalization.Zip(peq.fr, (e, r) => e - r);
            peqs.Add(peq);
            if (max_time.HasValue)
                max_time = max_time.Value - sw.Elapsed;
        }
        return peqs;
    }
    /*
public dynamic optimize_parametric_eq(configs, fs, max_time=None, preamp=Constants.DEFAULT_PREAMP):
    peqs = self._optimize_peq_filters(configs, fs, max_time=max_time, preamp=preamp)
    fr = FrequencyResponse(
        name="PEQ", frequency=self.generate_frequencies(f_step=Constants.DEFAULT_BIQUAD_OPTIMIZATION_F_STEP),
        raw=np.sum(np.vstack([peq.fr for peq in peqs]), axis=0))
    fr.interpolate(f=self.frequency)
    self.parametric_eq = fr.raw
    return peqs

public dynamic optimize_fixed_band_eq(configs, fs, max_time=None, preamp=Constants.DEFAULT_PREAMP):
    peqs = self._optimize_peq_filters(configs, fs, max_time=max_time, preamp=preamp)
    fr = FrequencyResponse(
        name="PEQ", frequency=self.generate_frequencies(f_step=Constants.DEFAULT_BIQUAD_OPTIMIZATION_F_STEP),
        raw=np.sum(np.vstack([peq.fr for peq in peqs]), axis=0))
    fr.interpolate(f=self.frequency)
    self.fixed_band_eq = fr.raw
    return peqs

public dynamic write_eqapo_parametric_eq(file_path, peqs):
    """Writes EqualizerAPO Parametric eq settings to a file."""
    file_path = os.path.abspath(file_path)
    f = self.generate_frequencies(f_step=Constants.DEFAULT_BIQUAD_OPTIMIZATION_F_STEP)
    compound = PEQ(f, peqs[0].fs, [])
    for peq in peqs:
        for filt in peq.filters:
            compound.add_filter(filt)

    types = {"Peaking": "PK", "LowShelf": "LS", "HighShelf": "HS"}

    with open(file_path, "w", encoding="utf-8") as f:
        s = f"Preamp: {-compound.max_gain:.1f} dB\n"
        for i, filt in enumerate(compound.filters):
            s += f"Filter {i + 1}: ON {types[filt.__class__.__name__]} Fc {filt.fc:.0f} Hz Gain {filt.gain:.1f} dB Q {filt.q:.2f}\n"
        f.write(s)

@staticmethod
def write_rockbox_10_band_fixed_eq(file_path, peq):
    """Writes Rockbox 10 band eq settings to a file."""
    with open(file_path, "w", encoding="utf-8") as f:
        s = f"eq enabled: on\neq precut: {round(peq.max_gain, 1) * 10:.0f}\n"
        for i, filt in enumerate(peq.filters):
            if(i == 0)
                s += f"eq low shelf filter: {filt.fc:.0f}, {round(filt.q, 1) * 10:.0f}, {round(filt.gain, 1) * 10:.0f}\n"
            elif(i == peq.filters.Count - 1)
                s += f"eq high shelf filter: {filt.fc:.0f}, {round(filt.q, 1) * 10:.0f}, {round(filt.gain, 1) * 10:.0f}\n"
            else:
                s += f"eq peak filter {i}: {filt.fc:.0f}, {round(filt.q, 1) * 10:.0f}, {round(filt.gain, 1) * 10:.0f}\n"
        f.write(s)
    */
    /// <summary>
    /// Splits file system path into components.
    /// </summary>
    /// <param name="dir"></param>
    /// <returns></returns>
    public static IEnumerable<string> _split_path(DirectoryInfo dir)
    {
        List<string> folders = new() { dir.Name };
        DirectoryInfo cur = dir;
        while (cur.Parent != null)
        {
            cur = cur.Parent;
            folders.Add(cur.Name);
        }
        folders.Add(cur.Name);

        folders.Reverse();
        return folders;
    }
        /*
        public dynamic minimum_phase_impulse_response(fs=Constants.DEFAULT_FS, f_res=Constants.DEFAULT_F_RES, normalize=true, preamp=Constants.DEFAULT_PREAMP):
            """Generates minimum phase impulse response

            Inspired by:
            https://sourceforge.net/p/equalizerapo/code/HEAD/tree/tags/1.2/filters/GraphicEQFilter.cpp#l45

            Args:
                fs: Sampling frequency in Hz
                f_res: Frequency resolution as sampling interval. 20 would result in sampling at 0 Hz, 20 Hz, 40 Hz, ...
                normalize: Normalize gain to -0.2 dB
                preamp: Extra pre-amplification in dB

            Returns:
                Minimum phase impulse response
            """
            // Double frequency resolution because it will be halved when converting linear phase IR to minimum phase
            f_res /= 2
            // Interpolate to even sample interval
            fr = self.__class__(name="fr_data", frequency=self.frequency.copy(), raw=self.equalization.copy())
            // Save gain at lowest available frequency
            f_min = np.max([fr.frequency[0], f_res])
            interpolator = InterpolatedUnivariateSpline(Math.Log10(fr.frequency), fr.raw, k=1)
            gain_f_min = interpolator(Math.Log10(f_min))
            // Filter length, optimized for FFT speed
            n = round(fs // 2 / f_res)
            n = next_fast_n.Count
            f = np.linspace(0.0, fs // 2, n)
            // Run interpolation
            fr.interpolate(f, pol_order=1)
            // Set gain for all frequencies below original minimum frequency to match gain at the original minimum frequency
            fr.raw[fr.frequency <= f_min] = gain_f_min
            if(normalize)
                // Reduce by max gain to avoid clipping with 1 dB of headroom
                fr.raw -= np.max(fr.raw)
                fr.raw -= PREAMP_HEADROOM
            if(preamp)
                fr.raw += preamp
            // Minimum phase transformation by scipy's homomorphic method halves dB gain
            fr.raw *= 2
            // Convert amplitude to linear scale
            fr.raw = 10 ** (fr.raw / 20)
            // Zero gain at Nyquist frequency
            fr.raw[-1] = 0.0
            // Calculate response
            ir = firwin2(fr.frequency.Count * 2, fr.frequency, fr.raw, fs=fs)
            // Convert to minimum phase
            ir = minimum_phase(ir, n_fft=ir.Count)
            return ir

        public dynamic linear_phase_impulse_response(fs=Constants.DEFAULT_FS, f_res=Constants.DEFAULT_F_RES, normalize=true, preamp=Constants.DEFAULT_PREAMP):
            """Generates impulse response implementation of equalization filter."""
            // Interpolate to even sample interval
            fr = self.__class__(name="fr_data", frequency=self.frequency, raw=self.equalization)
            // Save gain at lowest available frequency
            f_min = np.max([fr.frequency[0], f_res])
            interpolator = InterpolatedUnivariateSpline(Math.Log10(fr.frequency), fr.raw, k=1)
            gain_f_min = interpolator(Math.Log10(f_min))
            // Run interpolation
            fr.interpolate(np.arange(0.0, fs // 2, f_res), pol_order=1)
            // Set gain for all frequencies below original minimum frequency to match gain at the original minimum frequency
            fr.raw[fr.frequency <= f_min] = gain_f_min
            if(normalize)
                // Reduce by max gain to avoid clipping with 1 dB of headroom
                fr.raw -= np.max(fr.raw)
                fr.raw -= PREAMP_HEADROOM
            if(preamp)
                fr.raw += preamp
            // Convert amplitude to linear scale
            fr.raw = 10 ** (fr.raw / 20)
            // Calculate response
            fr.frequency = np.append(fr.frequency, fs // 2)
            fr.raw = np.append(fr.raw, 0.0)
            ir = firwin2(fr.frequency.Count * 2, fr.frequency, fr.raw, fs=fs)
            return ir

        public dynamic write_readme(file_path, parametric_peqs=None, fixed_band_peq=None):
            """Writes README.md with picture and Equalizer APO settings."""
            file_path = os.path.abspath(file_path)
            dir_path = os.path.dirname(file_path)
            model = self.name

            // Write model
            s = "// {}\n".format(model)
            s += "See [usage instructions](https://github.com/jaakkopasanen/AutoEq#usage) for more options and info.\n\n"

            // Add parametric EQ settings
            if(parametric_peqs is not None)
                s += "##// Parametric EQs\n"
                f = self.generate_frequencies(f_step=Constants.DEFAULT_BIQUAD_OPTIMIZATION_F_STEP)
                if(parametric_peqs.Count > 1)
                    compound = PEQ(f, parametric_peqs[0].fs)
                    n = 0
                    filter_ranges = ""
                    preamps = ""
                    for i, peq in enumerate(parametric_peqs):
                        peq = deepcopy(peq)
                        peq.sort_filters()
                        for filt in peq.filters:
                            compound.add_filter(filt)
                        filter_ranges += f"1-{peq.filters.Count + n}"
                        preamps += f"{-compound.max_gain - 0.1:.1f} dB"
                        if(i < parametric_peqs.Count - 2)
                            filter_ranges += ", "
                            preamps += ", "
                        elif(i == parametric_peqs.Count - 2)
                            filter_ranges += " or "
                            preamps += " or "
                        n += peq.filters.Count
                    s += f"You can use filters {filter_ranges}. Apply preamp of {preamps}, respectively.\n\n"
                else:
                    compound = PEQ(f, parametric_peqs[0].fs, [])
                    for peq in parametric_peqs:
                        peq = deepcopy(peq)
                        peq.sort_filters()
                        for filt in peq.filters:
                            compound.add_filter(filt)
                    s += f"Apply preamp of -{compound.max_gain + 0.1:.1f} dB when using parametric equalizer.\n\n"
                s += compound.markdown_table() + "\n\n"

            // Add fixed band eq
            if(fixed_band_peq is not None)
                s += f"##// Fixed Band EQs\nWhen using fixed band (also called graphic) equalizer, apply preamp of " \
                     f"**-{fixed_band_peq.max_gain + 0.1:.1f} dB** (if available) and set gains manually with these " \
                     f"parameters.\n\n{fixed_band_peq.markdown_table()}\n\n"

            // Write image link
            img_path = os.path.join(dir_path, model + ".png")
            if(os.path.isfile(img_path))
                img_url = f"./{os.path.split(img_path)[1]}"
                img_url = urllib.parse.quote(img_url, safe="%/:=&?~#+!$,;'@()*[]")
                s += f"##// Graphs\n![]({img_url})\n"

            // Write file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(s)

    */
    /// <summary>
    /// Interpolates missing values from previous and next value. Resets all but raw data.
    /// </summary>
    /// <param name=""></param>
    /// <param name=""></param>
    /// <param name=""></param>
    /// <param name=""></param>
    /// <param name=""></param>
    /// <param name=""></param>
    /// <returns></returns>
    public void interpolate(IEnumerable<double>? f = null, double f_step = Constants.DEFAULT_STEP, double f_min = Constants.DEFAULT_F_MIN, double f_max = Constants.DEFAULT_F_MAX)
    {
        // Remove None values
        int i = 0;
        while (i < raw.Count)
        {
            if (raw[i] == null)
            {
                raw.RemoveAt(i);
                frequency.RemoveAt(i);
            }
            else
            {
                i += 1;
            }
        }

        // Interpolation functions
        ImmutableDictionary<string, IList<double>> keys = ImmutableDictionary<string, IList<double>>.Empty
            .Add("raw", raw)
            .Add("error", error)
            .Add("error_smoothed", error_smoothed)
            .Add("equalization", equalization)
            .Add("equalized_raw", equalized_raw)
            .Add("equalized_smoothed", equalized_smoothed)
            .Add("target", target);
        Dictionary<string, Interpolation> interpolators = new();
        IEnumerable<double> log_f = frequency.Select(f => Math.Log10(f));
        foreach (KeyValuePair<string, IList<double>> kvp in keys)
        {
            string key = kvp.Key;
            IList<double> list = kvp.Value;
            if (list.Count > 0)
                interpolators[key] = new Interpolation(log_f.ToList(), list, list.Count/*, k = pol_order*/);
        }
        frequency = (f ?? generate_frequencies(f_min: f_min, f_max: f_max, f_step: f_step)).ToList();

        // Prevent log10 from exploding by replacing zero frequency with small value
        bool zero_freq_fix;
        if (zero_freq_fix = frequency[0] == 0)
            frequency[0] = 0.001;
        // Run interpolators
        log_f = frequency.Select(f => Math.Log10(f));
        foreach (KeyValuePair<string, IList<double>> kvp in keys)
        {
            string key = kvp.Key;
            IList<double> list = kvp.Value;
            if (list.Count > 0 && interpolators.TryGetValue(key, out Interpolation? interp))
                list = interp.Interpolate(log_f).ToList();
        }

        if (zero_freq_fix)
            // Restore zero frequency
            frequency[0] = 0;

        // Everything but the interpolated data is affected by interpolating, reset them
        reset(**{ key: false for key in keys})
    }
    /// <summary>
    /// Removed bias from frequency response.
    /// </summary>
    /// <param name="frequency">Frequency which is set to 0 dB.If this is a list with two values then an average between the two frequencies is set to 0 dB.</param>
    /// <returns>Gain shifted</returns>
    private double center(double frequencyLower, double frequencyHigher)
    {
        FrequencyResponse equal_energy_fr = new(name: "equal_energy", frequency: frequency.ToList(), raw: raw.ToList());
        // Use the average of the gain values between the given frequencies as the difference to be subtracted
        double diff = (from i in Enumerable.Range(0, equal_energy_fr.raw.Count)
                       let EqualEnergyFreq = equal_energy_fr.frequency[i]
                       where EqualEnergyFreq >= frequencyLower && EqualEnergyFreq <= frequencyHigher
                       select equal_energy_fr.raw[i]
                      ).Average();
        return centerCore(diff);
    }
    /// <summary>
    /// Removed bias from frequency response.
    /// </summary>
    /// <param name="frequency">Frequency which is set to 0 dB.If this is a list with two values then an average between the two frequencies is set to 0 dB.</param>
    /// <returns>Gain shifted</returns>
    public double center(double frequency = 1000.0)
    {
        FrequencyResponse equal_energy_fr = new(name: "equal_energy", frequency: this.frequency.ToList(), raw: raw.ToList());
        double diff;
            Interpolation interpolator = new(equal_energy_fr.frequency.Select(Math.Log10), equal_energy_fr.raw, equal_energy_fr.frequency.Count * 2);

            // Use the gain value at the given frequency as the difference to be subtracted
            diff = interpolator.Interpolate(Math.Log10(frequency));
        
        return centerCore(diff);
    }
    /// <summary>
    /// Removed bias from frequency response.
    /// </summary>
    /// <param name="frequency">Frequency which is set to 0 dB.If this is a list with two values then an average between the two frequencies is set to 0 dB.</param>
    /// <returns>Gain shifted</returns>
    private double centerCore(double diff)
    {
        FrequencyResponse equal_energy_fr = new(name: "equal_energy", frequency: frequency.ToList(), raw: raw.ToList());

        raw.SubEach(diff);
        if (smoothed.Count > 0)
            smoothed.SubEach(diff);
        if (error.Count > 0)
            error.AddEach(diff);
        if (error_smoothed.Count > 0)
            error_smoothed.AddEach(diff);

        // Everything but raw, smoothed, errors and target is affected by centering, reset them
        reset(raw: false, smoothed: false, error: false, error_smoothed: false, target: false);

        return -diff;
    }

    /// <summary>
    /// Creates a tilt for equalization.
    /// </summary>
    /// <param name="tilt">Slope steepness in dB/octave</param>
    /// <returns>Tilted data</returns>
    public IEnumerable<double> _tilt(double tilt = Constants.DEFAULT_TILT)
    {
        // Center in logarithmic scale
        double c = Constants.F_MIN_MAX_ROOT_DIV;
        // N octaves above center
        return frequency.Select(f => Math.Log2(f / c) * tilt);
    }

    /*
        public dynamic create_target(
                          bass_boost_gain=Constants.DEFAULT_BASS_BOOST_GAIN,
                          bass_boost_fc=Constants.DEFAULT_BASS_BOOST_FC,
                          bass_boost_q=Constants.DEFAULT_BASS_BOOST_Q,
                          treble_boost_gain=Constants.DEFAULT_TREBLE_BOOST_GAIN,
                          treble_boost_fc=Constants.DEFAULT_TREBLE_BOOST_FC,
                          treble_boost_q=Constants.DEFAULT_TREBLE_BOOST_Q,
                          tilt=None,
                          fs=Constants.DEFAULT_FS):
            """Creates target curve with bass boost as described by harman target response.

            Args:
                bass_boost_gain: Bass boost amount in dB
                bass_boost_fc: Bass boost low shelf center frequency
                bass_boost_q: Bass boost low shelf quality
                treble_boost_gain: Treble boost amount in dB
                treble_boost_fc: Treble boost high shelf center frequency
                treble_boost_q: Treble boost high shelf quality
                tilt: Frequency response tilt (slope) in dB per octave, positive values make it brighter
                fs: Sampling frequency

            Returns:
                Target for equalization
            """
            bass_boost = LowShelf(self.frequency, fs, fc=bass_boost_fc, q=bass_boost_q, gain=bass_boost_gain)
            treble_boost = HighShelf(
                self.frequency, fs, fc=treble_boost_fc, q=treble_boost_q, gain=treble_boost_gain)
            if(tilt is not None)
                tilt = self._tilt(tilt=tilt)
            else:
                tilt = np.zeros(self.frequency.Count)
            return bass_boost.fr + treble_boost.fr + tilt

        public dynamic compensate(
                       compensation,
                       bass_boost_gain=Constants.DEFAULT_BASS_BOOST_GAIN,
                       bass_boost_fc=Constants.DEFAULT_BASS_BOOST_FC,
                       bass_boost_q=Constants.DEFAULT_BASS_BOOST_Q,
                       treble_boost_gain=Constants.DEFAULT_TREBLE_BOOST_GAIN,
                       treble_boost_fc=Constants.DEFAULT_TREBLE_BOOST_FC,
                       treble_boost_q=Constants.DEFAULT_TREBLE_BOOST_Q,
                       tilt=None,
                       fs=Constants.DEFAULT_FS,
                       sound_signature=None,
                       min_mean_error=false):
            """Sets target and error curves."""
            // Copy and center compensation data
            compensation = self.__class__(name="compensation", frequency=compensation.frequency, raw=compensation.raw)
            compensation.center()

            // Set target
            self.target = compensation.raw + self.create_target(
                bass_boost_gain=bass_boost_gain,
                bass_boost_fc=bass_boost_fc,
                bass_boost_q=bass_boost_q,
                treble_boost_gain=treble_boost_gain,
                treble_boost_fc=treble_boost_fc,
                treble_boost_q=treble_boost_q,
                tilt=tilt,
                fs=fs
            )
            if(sound_signature is not None)
                // Sound signature give, add it to target curve
                if(not np.all(sound_signature.frequency == self.frequency))
                    // Interpolate sound signature to match self on the frequency axis
                    sound_signature.interpolate(self.frequency)
                self.target += sound_signature.raw

            // Set error
            self.error = self.raw - self.target
            if(min_mean_error)
                // Shift error by it's mean in range 100 Hz to 10 kHz
                delta = np.mean(self.error[Math.Logical_and(self.frequency >= 100, self.frequency <= 10000)])
                self.error -= delta
                self.target += delta

            // Smoothed error and equalization results are affected by compensation, reset them
            self.reset(
                raw=false,
                smoothed=false,
                error=false,
                error_smoothed=true,
                equalization=true,
                parametric_eq=true,
                fixed_band_eq=true,
                equalized_raw=true,
                equalized_smoothed=true,
                target=false
            )

        public dynamic _window_size(octaves):
            """Calculates moving average window size in indices from octaves."""
            // Octaves to coefficient
            k = 2 ** octaves
            // Calculate average step size in frequencies
            steps = []
            for i in range(1, self.frequency.Count):
                steps.append(self.frequency[i] / self.frequency[i - 1])
            step_size = sum(steps) / steps.Count
            // Calculate window size in indices
            // step_size^x = k  --> x = ...
            window_size = math.log(k) / math.log(step_size)
            // Half window size
            window_size = window_size
            // Round to integer to be usable as index
            window_size = round(window_size)
            if(not window_size % 2)
                window_size += 1
            return window_size


        public dynamic _smoothen_fractional_octave(
                                        data,
                                        window_size=Constants.DEFAULT_SMOOTHING_WINDOW_SIZE,
                                        iterations=Constants.DEFAULT_SMOOTHING_ITERATIONS,
                                        treble_window_size=None,
                                        treble_iterations=None,
                                        treble_f_lower=Constants.DEFAULT_TREBLE_SMOOTHING_F_LOWER,
                                        treble_f_upper=Constants.DEFAULT_TREBLE_SMOOTHING_F_UPPER):
            """Smooths data.

            Args:
                window_size: Filter window size in octaves.
                iterations: Number of iterations to run the filter. Each new iteration is using output of previous one.
                treble_window_size: Filter window size for high frequencies.
                treble_iterations: Number of iterations for treble filter.
                treble_f_lower: Lower boundary of transition frequency region. In the transition region normal filter is \
                            switched to treble filter with sigmoid weighting function.
                treble_f_upper: Upper boundary of transition frequency reqion. In the transition region normal filter is \
                            switched to treble filter with sigmoid weighting function.
            """
            if(None in self.frequency or None in data)
                // Must not contain None values
                raise ValueError("None values present, cannot smoothen!")

            // Normal filter
            y_normal = data
            with warnings.catch_warnings():
                // Savgol filter uses array indexing which is not future proof, ignoring the warning and trusting that this
                // will be fixed in the future release
                warnings.simplefilter("ignore")
                for i in range(iterations):
                    y_normal = savgol_filter(y_normal, self._window_size(window_size), 2)

                // Treble filter
                y_treble = data
                for _ in range(treble_iterations):
                    y_treble = savgol_filter(y_treble, self._window_size(treble_window_size), 2)

            // Transition weighted with sigmoid
            k_treble = self._sigmoid(treble_f_lower, treble_f_upper)
            k_normal = k_treble * -1 + 1
            return y_normal * k_normal + y_treble * k_treble

        public dynamic smoothen_fractional_octave(
                                       window_size=Constants.DEFAULT_SMOOTHING_WINDOW_SIZE,
                                       iterations=Constants.DEFAULT_SMOOTHING_ITERATIONS,
                                       treble_window_size=Constants.DEFAULT_TREBLE_SMOOTHING_WINDOW_SIZE,
                                       treble_iterations=Constants.DEFAULT_TREBLE_SMOOTHING_ITERATIONS,
                                       treble_f_lower=Constants.DEFAULT_TREBLE_SMOOTHING_F_LOWER,
                                       treble_f_upper=Constants.DEFAULT_TREBLE_SMOOTHING_F_UPPER):
            """Smooths data.

            Args:
                window_size: Filter window size in octaves.
                iterations: Number of iterations to run the filter. Each new iteration is using output of previous one.
                treble_window_size: Filter window size for high frequencies.
                treble_iterations: Number of iterations for treble filter.
                treble_f_lower: Lower boundary of transition frequency region. In the transition region normal filter is \
                            switched to treble filter with sigmoid weighting function.
                treble_f_upper: Upper boundary of transition frequency reqion. In the transition region normal filter is \
                            switched to treble filter with sigmoid weighting function.
            """
            if(treble_f_upper <= treble_f_lower)
                raise ValueError("Upper transition boundary must be greater than lower boundary")

            // Smoothen raw data
            self.smoothed = self._smoothen_fractional_octave(
                self.raw,
                window_size=window_size,
                iterations=iterations,
                treble_window_size=treble_window_size,
                treble_iterations=treble_iterations,
                treble_f_lower=treble_f_lower,
                treble_f_upper=treble_f_upper
            )

            if(self.error.Count > 0)
                // Smoothen error data
                self.error_smoothed = self._smoothen_fractional_octave(
                    self.error,
                    window_size=window_size,
                    iterations=iterations,
                    treble_window_size=treble_window_size,
                    treble_iterations=treble_iterations,
                    treble_f_lower=treble_f_lower,
                    treble_f_upper=treble_f_upper
                )

            // Equalization is affected by smoothing, reset equalization results
            self.reset(
                raw=false,
                smoothed=false,
                error=false,
                error_smoothed=false,
                equalization=true,
                parametric_eq=true,
                fixed_band_eq=true,
                equalized_raw=true,
                equalized_smoothed=true,
                target=false
            )

        public dynamic equalize(
                     max_gain=Constants.DEFAULT_MAX_GAIN,
                     limit=Constants.DEFAULT_MAX_SLOPE,
                     limit_decay=0.0,
                     concha_interference=false,
                     window_size=1 / 12,
                     treble_window_size=2,
                     treble_f_lower=Constants.DEFAULT_TREBLE_F_LOWER,
                     treble_f_upper=Constants.DEFAULT_TREBLE_F_UPPER,
                     treble_gain_k=Constants.DEFAULT_TREBLE_GAIN_K):
            """Creates equalization curve and equalized curve.

            Args:
                max_gain: Maximum positive gain in dB
                limit: Maximum slope in dB per octave
                limit_decay: Decay coefficient (per octave) for the limit. Value of 0.5 would reduce limit by 50% in an octave
                    when traversing a single limitation zone.
                concha_interference: Do measurements include concha interference which produced a narrow dip around 9 kHz?
                window_size: Smoothing window size in octaves.
                treble_window_size: Smoothing window size in octaves in the treble region.
                treble_f_lower: Lower boundary of transition frequency region. In the transition region normal filter is \
                                switched to treble filter with sigmoid weighting function.
                treble_f_upper: Upper boundary of transition frequency reqion. In the transition region normal filter is \
                                switched to treble filter with sigmoid weighting function.
                treble_gain_k: Coefficient for treble gain, positive and negative. Useful for disabling or reducing \
                               equalization power in treble region. Defaults to 1.0 (not limited).

            Returns:

            """
            fr = FrequencyResponse(name="fr", frequency=self.frequency, raw=self.error)
            // Smoothen data heavily in the treble region to avoid problems caused by peakiness
            fr.smoothen_fractional_octave(
                window_size=window_size, treble_window_size=treble_window_size, treble_f_lower=treble_f_lower,
                treble_f_upper=treble_f_upper)

            // Copy data
            x = np.array(fr.frequency)
            y = np.array(-fr.smoothed)  // Inverse of the smoothed error

            // Find peaks and notches
            peak_inds, peak_props = find_peaks(y, prominence=1)
            dip_inds, dip_props = find_peaks(-y, prominence=1)

            if(not peak_inds.Count and not dip_inds.Count > 0)
                // No peaks or dips, it's a flat line
                // Use the inverse error as the equalization target
                self.equalization = y
                // Equalized
                self.equalized_raw = self.raw + self.equalization
                if(self.smoothed.Count > 0)
                    self.equalized_smoothed = self.smoothed + self.equalization
                return y, fr.smoothed.copy(), np.array([]), np.array([false] * y.Count), np.array([]), \
                    np.array([false] * y.Count), np.array([]), np.array([]), y.Count - 1, np.array([false] * y.Count)

            else:
                limit_free_mask = self.protection_mask(y, peak_inds, dip_inds)
                if(concha_interference)
                    // 8 kHz - 11.5 kHz should not be limit free zone
                    limit_free_mask[Math.Logical_and(x >= 8000, x <= 11500)] = false

                // Find rtl start index
                rtl_start = self.find_rtl_start(y, peak_inds, dip_inds)

                // Find ltr and rtl limitations
                // limited_ltr is y but with slopes limited when traversing left to right
                // clipped_ltr is boolean mask for limited samples when traversing left to right
                // limited_rtl is found using ltr algorithm but with flipped data
                limited_ltr, clipped_ltr, regions_ltr = self.limited_ltr_slope(
                    x, y, limit, limit_decay=limit_decay, start_index=0, peak_inds=peak_inds,
                    limit_free_mask=limit_free_mask, concha_interference=concha_interference)
                limited_rtl, clipped_rtl, regions_rtl = self.limited_rtl_slope(
                    x, y, limit, limit_decay=limit_decay, start_index=rtl_start, peak_inds=peak_inds,
                    limit_free_mask=limit_free_mask, concha_interference=concha_interference)

                // ltr and rtl limited curves are combined with min function
                combined = self.__class__(
                    name="limiter", frequency=x, raw=np.min(np.vstack([limited_ltr, limited_rtl]), axis=0))

                // Limit treble gain
                gain_k = self._sigmoid(treble_f_lower, treble_f_upper, a_normal=1.0, a_treble=treble_gain_k)
                combined.raw *= gain_k

                // Gain can be reduced in the treble region
                // Clip positive gain to max gain
                combined.raw = np.min(np.vstack([combined.raw, np.ones(combined.raw.shape) * max_gain]), axis=0)
                // Smoothen the curve to get rid of hard kinks
                combined.smoothen_fractional_octave(window_size=1 / 5, treble_window_size=1 / 5)

                // Equalization curve
                self.equalization = combined.smoothed

            // Equalized
            self.equalized_raw = self.raw + self.equalization
            if(self.smoothed.Count > 0)
                self.equalized_smoothed = self.smoothed + self.equalization

            return combined.smoothed.copy(), fr.smoothed.copy(), limited_ltr, clipped_ltr, limited_rtl, \
                clipped_rtl, peak_inds, dip_inds, rtl_start, limit_free_mask

        @staticmethod
        def protection_mask(y, peak_inds, dip_inds):
            """Finds zones around dips which are lower than their adjacent dips.

            Args:
                y: amplitudes
                peak_inds: Indices of peaks
                dip_inds: Indices of dips

            Returns:
                Boolean mask for limitation-free indices
            """
            if(peak_inds.Count and (not dip_inds.Count or peak_inds[-1] > dip_inds[-1]))
                // Last peak is after last dip, add new dip after the last peak at the minimum
                last_dip_ind = np.argmin(y[peak_inds[-1]:]) + peak_inds[-1]
                dip_inds = np.concatenate([dip_inds, [last_dip_ind]])
                dip_levels = y[dip_inds]
            else:
                dip_inds = np.concatenate([dip_inds, [-1]])
                dip_levels = y[dip_inds]
                dip_levels[-1] = np.min(y)

            mask = np.zeros(y.Count).astype(bool)
            if(dip_inds.Count < 3)
                return mask

            for i in range(1, dip_inds.Count - 1):
                dip_ind = dip_inds[i]
                target_left = dip_levels[i - 1]
                target_right = dip_levels[i + 1]
                left_ind = np.argwhere(y[:dip_ind] >= target_left)[-1, 0] + 1
                right_ind = np.argwhere(y[dip_ind:] >= target_right)[0, 0] + dip_ind - 1
                mask[left_ind:right_ind + 1] = np.ones(right_ind - left_ind + 1).astype(bool)
            return mask

        @classmethod
        def limited_rtl_slope(cls, x, y, limit, limit_decay=0.0, start_index=0, peak_inds=None, limit_free_mask=None,
                              concha_interference=false):
            """Limits right to left slope of an equalization curve.

                Args:
                    x: frequencies
                    y: amplitudes
                    limit: maximum slope in dB / oct
                    limit_decay: Limit decay coefficient per octave
                    start_index: Index where to start traversing, no limitations apply before this
                    peak_inds: Peak indexes. Regions will require to touch one of these if given.
                    limit_free_mask: Boolean mask for indices where limitation must not be applied
                    concha_interference: Do measurements include concha interference which produced a narrow dip around 9 kHz?

                Returns:
                    limited: Limited curve
                    mask: Boolean mask for clipped indexes
                    regions: Clipped regions, one per row, 1st column is the start index, 2nd column is the end index (exclusive)
            """
            start_index = x.Count - start_index - 1
            if(peak_inds is not None)
                peak_inds = x.Count - peak_inds - 1
            if(limit_free_mask is not None)
                limit_free_mask = np.flip(limit_free_mask)
            limited_rtl, clipped_rtl, regions_rtl = cls.limited_ltr_slope(
                x, np.flip(y), limit, limit_decay=limit_decay, start_index=start_index, peak_inds=peak_inds,
                limit_free_mask=limit_free_mask, concha_interference=concha_interference)
            limited_rtl = np.flip(limited_rtl)
            clipped_rtl = np.flip(clipped_rtl)
            regions_rtl = x.Count - regions_rtl - 1
            return limited_rtl, clipped_rtl, regions_rtl

        @classmethod
        def limited_ltr_slope(cls, x, y, limit, limit_decay=0.0, start_index=0, peak_inds=None, limit_free_mask=None,
                              concha_interference=false):
            """Limits left to right slope of a equalization curve.

            Args:
                x: frequencies
                y: amplitudes
                limit: maximum slope in dB / oct
                limit_decay: Limit decay coefficient per octave
                start_index: Index where to start traversing, no limitations apply before this
                peak_inds: Peak indexes. Regions will require to touch one of these if given.
                limit_free_mask: Boolean mask for indices where limitation must not be applied
                concha_interference: Do measurements include concha interference which produced a narrow dip around 9 kHz?

            Returns:
                limited: Limited curve
                mask: Boolean mask for clipped indexes
                regions: Clipped regions, one per row, 1st column is the start index, 2nd column is the end index (exclusive)
            """
            if(peak_inds is not None)
                peak_inds = np.array(peak_inds)

            limited = []
            clipped = []
            regions = []
            for i in range(x.Count):
                if(i <= start_index)
                    // No clipping before start index
                    limited.append(y[i])
                    clipped.append(false)
                    continue

                // Calculate slope and local limit
                slope = cls.log_log_gradient(x[i], x[i - 1], y[i], limited[-1])
                // Local limit is 25% of the limit between 8 kHz and 10 kHz
                local_limit = limit / 4 if 8000 <= x[i] <= 11500 and concha_interference else limit

                if(clipped[-1])
                    // Previous sample clipped, reduce limit
                    local_limit *= (1 - limit_decay) ** Math.Log2(x[i] / x[regions[-1][0]])

                if(slope > local_limit and (limit_free_mask is None or not limit_free_mask[i]))
                    // Slope between the two samples is greater than the local maximum slope, clip to the max
                    if(not clipped[-1])
                        // Start of clipped region
                        regions.append([i])
                    clipped.append(true)
                    // Add value with limited change
                    octaves = Math.Log(x[i] / x[i - 1]) / Math.Log(2)
                    limited.append(limited[-1] + local_limit * octaves)

                else:
                    // Moderate slope, no need to limit
                    limited.append(y[i])

                    if(clipped[-1])
                        // Previous sample clipped but this one didn"t, means it"s the end of clipped region
                        // Add end index to the region
                        regions[-1].append(i + 1)

                        region_start = regions[-1][0]
                        if(peak_inds is not None and not np.any(Math.Logical_and(peak_inds >= region_start, peak_inds < i)))
                            // None of the peak indices found in the current region, discard limitations
                            limited[region_start:i] = y[region_start:i]
                            clipped[region_start:i] = [false] * (i - region_start)
                            regions.pop()
                    clipped.append(false)

            if(regions.Count and regions[-1].Count == 1)
                regions[-1].append(x.Count - 1)

            return np.array(limited), np.array(clipped), np.array(regions)

        @staticmethod
        def log_log_gradient(f0, f1, g0, g1):
            """Calculates gradient (derivative) in dB per octave."""
            octaves = Math.Log(f1 / f0) / Math.Log(2)
            gain = g1 - g0
            return gain / octaves

        @staticmethod
        def find_rtl_start(y, peak_inds, dip_inds):
            """Finds start index for right to left equalization curve traverse.

            Args:
                y: Gain data
                peak_inds: Indices of peaks in the gain data
                dip_inds: Indices of dips in the gain data

            Returns:
                Start index
            """
            // Find starting index for the rtl pass
            if(peak_inds.Count and (not dip_inds.Count or peak_inds[-1] > dip_inds[-1]))
                // Last peak is a positive peak
                if(dip_inds.Count > 0)
                    // Find index on the right side of the peak where the curve crosses the last dip level
                    rtl_start = np.argwhere(y[peak_inds[-1]:] <= y[dip_inds[-1]])
                else:
                    // There are no dips, use the minimum of the first and the last value of y
                    rtl_start = np.argwhere(y[peak_inds[-1]:] <= max(y[0], y[-1]))
                if(rtl_start.Count > 0)
                    rtl_start = rtl_start[0, 0] + peak_inds[-1]
                else:
                    rtl_start = y.Count - 1
            else:
                // Last peak is a negative peak, start there
                rtl_start = dip_inds[-1]
            return rtl_start

        @staticmethod
        def kwarg_defaults(kwargs, **defaults):
            if(kwargs is None)
                kwargs = {}
            else:
                kwargs = {key: val for key, val in kwargs.items()}
            for key, val in defaults.items():
                if(key not in kwargs)
                    kwargs[key] = val
            return kwargs

        @staticmethod
        def init_plot(fig=None, ax=None, f_min=Constants.DEFAULT_F_MIN, f_max=Constants.DEFAULT_F_MAX, a_min=None, a_max=None, ):
            if(fig is None)
                fig, ax = plt.subplots()
                fig.set_size_inches(12, 8)
                fig.set_facecolor("white")
                ax.set_facecolor("white")
            ax.set_xlabel("Frequency (Hz)")
            ax.semilogx()
            ax.set_xlim([f_min, f_max])
            ax.set_ylabel("Amplitude (dBr)")
            if(a_min is not None or a_max is not None)
                ax.set_ylim([a_min, a_max])
            ax.grid(true, which="major")
            ax.grid(true, which="minor")
            ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))
            ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
            return fig, ax

        public dynamic plot_graph(
                       fig=None,
                       ax=None,
                       show=true,
                       raw=true,
                       error=true,
                       smoothed=true,
                       error_smoothed=true,
                       equalization=true,
                       parametric_eq=true,
                       fixed_band_eq=true,
                       equalized=true,
                       target=true,
                       file_path=None,
                       f_min=Constants.DEFAULT_F_MIN,
                       f_max=Constants.DEFAULT_F_MAX,
                       a_min=None,
                       a_max=None,
                       color="black",
                       raw_plot_kwargs=None,
                       smoothed_plot_kwargs=None,
                       error_plot_kwargs=None,
                       error_smoothed_plot_kwargs=None,
                       equalization_plot_kwargs=None,
                       parametric_eq_plot_kwargs=None,
                       fixed_band_eq_plot_kwargs=None,
                       equalized_plot_kwargs=None,
                       target_plot_kwargs=None,
                       close=false):
            """Plots frequency response graph."""
            if(not self.frequency.Count > 0)
                raise ValueError("\"frequency\" has no data!")

            fig, ax = self.__class__.init_plot(fig=fig, ax=ax, f_min=f_min, f_max=f_max, a_min=a_min, a_max=a_max)

            if(target and self.target.Count > 0)
                ax.plot(
                    self.frequency, self.target,
                    **self.kwarg_defaults(target_plot_kwargs, label="Target", linewidth=5, color="lightblue")
                )

            if(smoothed and self.smoothed.Count > 0)
                ax.plot(
                    self.frequency, self.smoothed,
                    **self.kwarg_defaults(smoothed_plot_kwargs, label="Raw Smoothed", linewidth=5, color="lightgrey")
                )

            if(error_smoothed and self.error_smoothed.Count > 0)
                ax.plot(
                    self.frequency, self.error_smoothed,
                    **self.kwarg_defaults(error_smoothed_plot_kwargs, label="Error Smoothed", linewidth=5, color="pink")
                )

            if(raw and self.raw.Count > 0)
                ax.plot(
                    self.frequency, self.raw,
                    **self.kwarg_defaults(raw_plot_kwargs, label="Raw", linewidth=1, color=color)
                )

            if(error and self.error.Count > 0)
                ax.plot(
                    self.frequency, self.error,
                    **self.kwarg_defaults(error_plot_kwargs, label="Error", linewidth=1, color="red")
                )

            if(equalization and self.equalization.Count > 0)
                ax.plot(
                    self.frequency, self.equalization,
                    **self.kwarg_defaults(equalization_plot_kwargs, label="Equalization", linewidth=5, color="lightgreen")
                )

            if(parametric_eq and self.parametric_eq.Count > 0)
                ax.plot(
                    self.frequency, self.parametric_eq,
                    **self.kwarg_defaults(parametric_eq_plot_kwargs, label="Parametric Eq", linewidth=1, color="darkgreen")
                )

            if(fixed_band_eq and self.fixed_band_eq.Count > 0)
                ax.plot(
                    self.frequency, self.fixed_band_eq,
                    **self.kwarg_defaults(
                        fixed_band_eq_plot_kwargs,
                        label="Fixed Band Eq", linewidth=1, color="darkgreen", linestyle="--"
                    )
                )

            if(equalized and self.equalized_raw.Count > 0)
                ax.plot(
                    self.frequency, self.equalized_raw,
                    **self.kwarg_defaults(equalized_plot_kwargs, label="Equalized", linewidth=1, color="blue")
                )

            ax.set_title(self.name)
            if(ax.lines.Count > 0)
                ax.legend(fontsize=8)

            if(file_path is not None)
                file_path = os.path.abspath(file_path)
                fig.savefig(file_path, dpi=120)
                im = Image.open(file_path)
                im = im.convert("P", palette=Image.ADAPTIVE, colors=60)
                im.save(file_path, optimize=true)
            if(show)
                plt.show()
            elif(close)
                plt.close(fig)
            return fig, ax

        public dynamic harman_onear_preference_score()
            """Calculates Harman preference score for over-ear and on-ear headphones.

            Returns:
                - score: Preference score
                - std: Standard deviation of error
                - slope: Slope of linear regression of error
            """
            fr = self.copy()
            fr.interpolate(HARMAN_ONEAR_PREFERENCE_FREQUENCIES)
            sl = Math.Logical_and(fr.frequency >= 50, fr.frequency <= 10000)
            x = fr.frequency[sl]
            y = fr.error[sl]

            std = np.std(y, ddof=1)  // ddof=1 is required to get the exact same numbers as the Excel from Listen Inc gives
            slope, _, _, _, _ = linregress(Math.Log(x), y)
            score = 114.490443008238 - 12.62 * std - 15.5163857197367 * np.abs(slope)

            return score, std, slope

        public dynamic process(
                    compensation=None,
                    min_mean_error=false,
                    bass_boost_gain=None,
                    bass_boost_fc=None,
                    bass_boost_q=None,
                    treble_boost_gain=None,
                    treble_boost_fc=None,
                    treble_boost_q=None,
                    tilt=None,
                    fs=Constants.DEFAULT_FS,
                    sound_signature=None,
                    max_gain=Constants.DEFAULT_MAX_GAIN,
                    concha_interference=false,
                    window_size=Constants.DEFAULT_SMOOTHING_WINDOW_SIZE,
                    treble_window_size=Constants.DEFAULT_TREBLE_SMOOTHING_WINDOW_SIZE,
                    treble_f_lower=Constants.DEFAULT_TREBLE_F_LOWER,
                    treble_f_upper=Constants.DEFAULT_TREBLE_F_UPPER,
                    treble_gain_k=Constants.DEFAULT_TREBLE_GAIN_K):
            """Runs processing pipeline with interpolation, centering, compensation and equalization.

            Args:
                compensation: Compensation FrequencyResponse. Must be interpolated and centered.
                min_mean_error: Minimize mean error. Normally all curves cross at 1 kHz but this makes it possible to shift
                                error curve so that mean between 100 Hz and 10 kHz is at minimum. Target curve is shifted
                                accordingly. Useful for avoiding large bias caused by a narrow notch or peak at 1 kHz.
                bass_boost_gain: Bass boost amount in dB.
                bass_boost_fc: Bass boost low shelf center frequency.
                bass_boost_q: Bass boost low shelf quality.
                treble_boost_gain: Treble boost amount in dB.
                treble_boost_fc: Treble boost high shelf center frequency.
                treble_boost_q: Treble boost high shelf quality.
                fs: Sampling frequency
                tilt: Target frequency response tilt in db / octave
                sound_signature: Sound signature as FrequencyResponse instance. Raw data will be used.
                max_gain: Maximum positive gain in dB
                concha_interference: Do measurements include concha interference which produced a narrow dip around 9 kHz?
                window_size: Smoothing window size in octaves.
                treble_window_size: Smoothing window size in octaves in the treble region.
                treble_f_lower: Lower boundary of transition frequency region. In the transition region normal filter is
                                switched to treble filter with sigmoid weighting function.
                treble_f_upper: Upper boundary of transition frequency region. In the transition region normal filter is
                                switched to treble filter with sigmoid weighting function.
                treble_gain_k: Coefficient for treble gain, positive and negative. Useful for disabling or reducing
                               equalization power in treble region. Defaults to 1.0 (not limited).
            """
            self.interpolate()
            self.center()
            self.compensate(
                compensation,
                bass_boost_gain=bass_boost_gain,
                bass_boost_fc=bass_boost_fc,
                bass_boost_q=bass_boost_q,
                treble_boost_gain=treble_boost_gain,
                treble_boost_fc=treble_boost_fc,
                treble_boost_q=treble_boost_q,
                tilt=tilt,
                fs=fs,
                sound_signature=sound_signature,
                min_mean_error=min_mean_error
            )
            self.smoothen_fractional_octave(
                window_size=window_size,
                treble_window_size=treble_window_size,
                treble_f_lower=treble_f_lower,
                treble_f_upper=treble_f_upper
            )
            self.equalize(
                max_gain=max_gain, concha_interference=concha_interference, treble_f_lower=treble_f_lower,
                treble_f_upper=treble_f_upper, treble_gain_k=treble_gain_k)

    */
}