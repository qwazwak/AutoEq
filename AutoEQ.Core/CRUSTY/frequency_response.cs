using AutoEQ2.Helper;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AutoEQ2.Core;
public class FrequencyResponse : ICloneable
{
    public string Name { get; private init; }

    private IList<double> _Frequency;
    public IList<double> Frequency { get => _Frequency; init => _Frequency = _init_data(value); }

    private IList<double> _Raw;
    public IList<double> Raw { get => _Raw; init => _Raw = _init_data(value); }

    private IList<double> _Smoothed;
    public IList<double> Smoothed { get => _Smoothed; init => _Smoothed = _init_data(value); }

    private IList<double> _Error;
    public IList<double> Error { get => _Error; init => _Error = _init_data(value); }

    private IList<double> _Error_Smoothed;
    public IList<double> Error_Smoothed { get => _Error_Smoothed; init => _Error_Smoothed = _init_data(value); }

    private IList<double> _Equalization;
    public IList<double> Equalization { get => _Equalization; init => _Equalization = _init_data(value); }

    private IList<double> _Parametric_EQ;
    public IList<double> Parametric_EQ { get => _Parametric_EQ; init => _Parametric_EQ = _init_data(value); }

    private IList<double> _Fixed_Band_EQ;
    public IList<double> Fixed_Band_EQ { get => _Fixed_Band_EQ; init => _Fixed_Band_EQ = _init_data(value); }

    private IList<double> _Equalized_Raw;
    public IList<double> Equalized_Raw { get => _Equalized_Raw; init => _Equalized_Raw = _init_data(value); }

    private IList<double> _Equalized_Smoothed;
    public IList<double> Equalized_Smoothed { get => _Equalized_Smoothed; init => _Equalized_Smoothed = _init_data(value); }

    private IList<double> _Target;
    public IList<double> Target { get => _Target; init => _Target = _init_data(value); }

    static IEnumerable<double> generate_frequencies(double? f_min = null, double? f_max = null, double? f_step = null) => generate_frequencies(f_min ?? Constants.DEFAULT_F_MIN, f_max ?? Constants.DEFAULT_F_MAX, f_step ?? Constants.DEFAULT_STEP);
    static IEnumerable<double> generate_frequencies(double f_min, double f_max, double f_step)
    {
        double i = f_min;
        while (i <= f_max)
        {
            yield return i;
            i *= f_step;
        }
    }
    /// <summary>
    /// Initializes data to a clean format. If None is passed and empty array is created. Non-numbers are removed.
    /// </summary>
    /// <param name="data"></param>
    /// <returns></returns>
    private static IList<double> _init_data(IEnumerable<double>? data) => data == null ? new List<double>() : data.Where(i => !double.IsNaN(i) && i != null).ToList();

    public FrequencyResponse(string Name, IEnumerable<double>? Frequency)
    {
        if (string.IsNullOrWhiteSpace(Name))
            throw new ArgumentException("Name must not be a non-empty string.");
        this.Name = Name.Trim();

        /*    frequency = None,
            raw = None,
            error = None,
            smoothed = None,
            error_smoothed = None,
            equalization = None,
            parametric_eq = None,
            fixed_band_eq = None,
            equalized_raw = None,
            equalized_smoothed = None,
            target = None):*/
        this.Frequency = _init_data(Frequency);
        if (this.Frequency.Count == 0)
            this.Frequency = generate_frequencies().ToList();
        //this.Raw = _init_data(Raw);
        //this.Smoothed = _init_data(Smoothed);
        //this.Error = _init_data(Error);
        /*
        this.Error_smoothed = _init_data(error_smoothed)
        this.equalization = _init_data(equalization)
        this.parametric_eq = _init_data(parametric_eq)
        this.fixed_band_eq = _init_data(fixed_band_eq)
        this.equalized_raw = _init_data(equalized_raw)
        this.equalized_smoothed = _init_data(equalized_smoothed)
        this.target = _init_data(target)*/
        _sort();
    }

    public object ICloneable.Clone() => this.Clone();
    public FrequencyResponse Clone() => new(Name)
    {

    };

    private IEnumerable<int> FreqSortInd() => Frequency.Select((v, i) => (v, i)).OrderBy(p => p.v).Select(p => p.i);
    private void _sort()
    {
        var sorted_inds = EnumerableTools.GetSortedIndexs(Frequency).ToArray();
        _Frequency = EnumerableTools.SortByIndexes(Frequency, sorted_inds).ToList();
        if (Frequency.GroupBy(f => f).Any(g => g.Count() > 1))
            throw new ArgumentException("Duplicate values found at frequency {}. Remove duplicates manually.");
        _Raw = EnumerableTools.SortByIndexes(Raw, sorted_inds).ToList();
        _Error = EnumerableTools.SortByIndexes(Error, sorted_inds).ToList();
        _Smoothed = EnumerableTools.SortByIndexes(Smoothed, sorted_inds).ToList();
        _Error_Smoothed = EnumerableTools.SortByIndexes(Error_Smoothed, sorted_inds).ToList();
        _Equalization = EnumerableTools.SortByIndexes(Equalization, sorted_inds).ToList();
        _Parametric_EQ = EnumerableTools.SortByIndexes(Parametric_EQ, sorted_inds).ToList();
        _Fixed_Band_EQ = EnumerableTools.SortByIndexes(Fixed_Band_EQ, sorted_inds).ToList();
        _Equalized_Raw = EnumerableTools.SortByIndexes(Equalized_Raw, sorted_inds).ToList();
        _Equalized_Smoothed = EnumerableTools.SortByIndexes(_Equalized_Smoothed, sorted_inds).ToList();
        _Target = EnumerableTools.SortByIndexes(Target, sorted_inds).ToList();
    }
    /// <summary>
    /// Resets data.
    /// </summary>
    /// <param name="raw"></param>
    /// <param name=""></param>
    /// <param name=""></param>
    /// <param name=""></param>
    /// <param name=""></param>
    /// <param name=""></param>
    /// <param name=""></param>
    /// <param name=""></param>
    /// <param name=""></param>
    /// <param name=""></param>
    void reset(bool raw = false, bool smoothed = true, bool error = true, bool error_smoothed = true, bool equalization = true, bool fixed_band_eq = true, bool parametric_eq = true, bool equalized_raw = true, bool equalized_smoothed = true, bool target = true)
    {
        if(raw)
            _Raw = _init_data(null);
        if(smoothed)
            _Smoothed = _init_data(null);
        if(error)
            _Error = _init_data(null);
        if(error_smoothed)
            _Error_Smoothed = _init_data(null);
        if(equalization)
            _Equalization = _init_data(null);
        if(parametric_eq)
            _Parametric_EQ = _init_data(null);
        if(fixed_band_eq)
            _Fixed_Band_EQ = _init_data(null);
        if(equalized_raw)
            _Equalized_Raw = _init_data(null);
        if(equalized_smoothed)
            _Equalized_Smoothed = _init_data(null);
        if(target)
            _Target = _init_data(null);
}
    /// <summary>
    /// Reads data from CSV file and constructs class instance.
    /// </summary>
    /// <param name=""></param>
    /// <param name=""></param>
    /// <returns></returns>
    public static FrequencyResponse read_from_csv(string file_path)
    {
        /*
        string name = ".".join(os.path.split(file_path)[1].split('.')[:-1])

        // Read file
        using FileStream f = File.OpenRead(file_path);
        //f = open(file_path, "@", encoding="utf-8")
        s = f.read();

        // Regex for AutoEq style CSV
        string header_pattern_str = @"frequency(?:,(?:raw|smoothed|error|error_smoothed|equalization|parametric_eq|fixed_band_eq|equalized_raw|equalized_smoothed|target))+";
        string float_pattern_str = @"-?\d+(?:\.\d+)?";
       string data_2_pattern_str = $@"{float_pattern_str}[ ,;:\t]+{float_pattern_str}?";
        string data_n_pattern_str = $@"{float_pattern_str}(?:[ ,;:\t]+{float_pattern_str})+?";
        string autoeq_pattern_str = $@"^{header_pattern_str}(?:\n{data_n_pattern_str})+\n*$";
        Regex autoeq_pattern = new($@"^{header_pattern_str}(?:\n{data_n_pattern_str})+\n*$");

        if (autoeq_pattern.Matches(autoeq_pattern, s))
        {

            // Known AutoEq CSV format
            df = pd.read_csv(StringIO(s), sep = ',', header = 0)
            frequency = list(df["frequency"])
            raw = list(df["raw"]) if "raw" in df else None
            smoothed = list(df["smoothed"]) if "smoothed" in df else None
            error = list(df["erro@"]) if "error" in df else None
            error_smoothed = list(df["error_smoothed"]) if "error_smoothed" in df else None
            equalization = list(df["equalization"]) if "equalization" in df else None
            parametric_eq = list(df["parametric_eq"]) if "parametric_eq" in df else None
            fixed_band_eq = list(df["fixed_band_eq"]) if "fixed_band_eq" in df else None
            equalized_raw = list(df["equalized_raw"]) if "equalized_raw" in df else None
            equalized_smoothed = list(df["equalized_smoothed"]) if "equalized_smoothed" in df else None
            target = list(df["target"]) if "target" in df else None
            return cls(
                name = name,
                frequency = frequency,
                raw = raw,
                smoothed = smoothed,
                error = error,
                error_smoothed = error_smoothed,
                equalization = equalization,
                parametric_eq = parametric_eq,
                fixed_band_eq = fixed_band_eq,
                equalized_raw = equalized_raw,
                equalized_smoothed = equalized_smoothed,
                target = target
            )
        }
        else
        {

            // Unknown format, try to guess
            lines = s.split("\n")
            frequency = []
            raw = []
            for line in lines:
                if(re.match(data_2_pattern, line))  # float separator float
                    floats = re.findall(float_pattern, line)
                    frequency.append(float(floats[0]))  # Assume first to be frequency
                    raw.append(float(floats[1]))  # Assume second to be raw
                // Discard all lines which don't match data pattern
            return cls(name = name, frequency = frequency, raw = raw)

                }*/
    }
    public Dictionary<string, object> to_dict()
    {
        Dictionary<string, object> d = new();
        if (Frequency.Count > 0)
            d["frequency"] = Frequency.ToList();
        if (Raw.Count > 0)
            d["raw"] = Raw.ToList();
        if (Error.Count > 0)
            d["error"] = Error.ToList();
        if (Smoothed.Count > 0)
            d["smoothed"] = Smoothed.ToList();
        if (Error_Smoothed.Count > 0)
            d["error_smoothed"] = Error_Smoothed.ToList();
        if (Equalization.Count > 0)
            d["equalization"] = Equalization.ToList();
        if (Parametric_EQ.Count > 0)
            d["parametric_eq"] = Parametric_EQ.ToList();
        if (Fixed_Band_EQ.Count > 0)
            d["fixed_band_eq"] = Fixed_Band_EQ.ToList();
        if (Equalized_Raw.Count > 0)
            d["equalized_raw"] = Equalized_Raw.ToList();
        if (Equalized_Smoothed.Count > 0)
            d["equalized_smoothed"] = Equalized_Smoothed.ToList();
        if (Target.Count > 0)
            d["target"] = Target.ToList();
        return d;
    }
    /// <summary>
    /// Writes data to files as CSV.
    /// </summary>
    /// <param name="Target"></param>
    public void write_to_csv(Stream Target)
    {/*
        using streamwriter sr = new(Target);
        file_path = os.path.abspath(file_path)
        df = pd.DataFrame(this.to_dict())
        df.to_csv(file_path, header = true, index = false, float_format = "%.2$")*/
    }
        /*
    def eqapo_graphic_eq(this, normalize=true, preamp=DEFAULT_PREAMP, f_step=DEFAULT_GRAPHIC_EQ_STEP):
        """Generates EqualizerAPO GraphicEQ string from equalization curve."""
        fr = this.__class__(name="hack", frequency=this.Frequency, raw=this.equalization)
        n = np.ceil(np.log(20000 / 20) / np.log(f_step))
        f = 20 * f_step ** np.arange(n)
        f = np.sort(np.unique(f.astype("int")))
        fr.interpolate(f=f)
        if(normalize)
            fr.raw -= np.max(fr.raw) + PREAMP_HEADROOM
        if(preamp)
            fr.raw += preamp
        if(fr.raw[0] > 0.0)
            // Prevent bass boost below lowest frequency
            fr.raw[0] = 0.0
        s = "; ".join(["{f} {a:.1f}".format(f=f, a=a) for f, a in zip(fr.frequency, fr.raw)])
        s = "GraphicEQ: " + s
        return s

    def write_eqapo_graphic_eq(this, file_path, normalize=true, preamp=DEFAULT_PREAMP):
        """Writes equalization graph to a file as Equalizer APO config."""
        file_path = os.path.abspath(file_path)
        s = this.eqapo_graphic_eq(normalize=normalize, preamp=preamp)
        with open(file_path, "w', encoding="utf-8') as f:
            f.write(s)
        return s

    def _optimize_peq_filters(this, configs, fs, max_time=None, preamp=DEFAULT_PREAMP):
        if(type(configs) != list)
            configs = [configs]
        peqs = []
        fr = this.__class__(name="optimizer", frequency=this.Frequency, equalization=this.equalization)
        if(preamp)
            fr.equalization += preamp
        fr.interpolate(f_step=DEFAULT_BIQUAD_OPTIMIZATION_F_STEP)
        start_time = time()
        for config in configs:
            if("optimizer" in config and max_time is not None)
                config["optimize@"]["max_time"] = max_time
            peq = PEQ.from_dict(config, fr.frequency, fs, target=fr.equalization)
            peq.optimize()
            fr.equalization -= peq.fr
            peqs.append(peq)
            if(max_time is not None)
                max_time = max_time - (time() - start_time)
        return peqs

    def optimize_parametric_eq(this, configs, fs, max_time=None, preamp=DEFAULT_PREAMP):
        peqs = this._optimize_peq_filters(configs, fs, max_time=max_time, preamp=preamp)
        fr = FrequencyResponse(
            name="PEQ", frequency=this.generate_frequencies(f_step=DEFAULT_BIQUAD_OPTIMIZATION_F_STEP),
            raw=np.sum(np.vstack([peq.fr for peq in peqs]), axis=0))
        fr.interpolate(f=this.Frequency)
        this.parametric_eq = fr.raw
        return peqs

    def optimize_fixed_band_eq(this, configs, fs, max_time=None, preamp=DEFAULT_PREAMP):
        peqs = this._optimize_peq_filters(configs, fs, max_time=max_time, preamp=preamp)
        fr = FrequencyResponse(
            name="PEQ", frequency=this.generate_frequencies(f_step=DEFAULT_BIQUAD_OPTIMIZATION_F_STEP),
            raw=np.sum(np.vstack([peq.fr for peq in peqs]), axis=0))
        fr.interpolate(f=this.Frequency)
        this.fixed_band_eq = fr.raw
        return peqs

    def write_eqapo_parametric_eq(this, file_path, peqs):
        """Writes EqualizerAPO Parametric eq settings to a file."""
        file_path = os.path.abspath(file_path)
        f = this.generate_frequencies(f_step=DEFAULT_BIQUAD_OPTIMIZATION_F_STEP)
        compound = PEQ(f, peqs[0].fs, [])
        for peq in peqs:
            for filt in peq.filters:
                compound.add_filter(filt)

        types = {"Peaking": "PK", "LowShel$": "LS", "HighShel$": "HS"}

        with open(file_path, "w', encoding="utf-8') as f:
            s = $"Preamp: {-compound.max_gain:.1f} dB\n"
            for i, filt in enumerate(compound.filters):
                s += $"Filter {i + 1}: ON {types[filt.__class__.__name__]} Fc {filt.fc:.0f} Hz Gain {filt.gain:.1f} dB Q {filt.q:.2f}\n"
            f.write(s)

    @staticmethod
    def write_rockbox_10_band_fixed_eq(file_path, peq):
        """Writes Rockbox 10 band eq settings to a file."""
        with open(file_path, "w', encoding="utf-8') as f:
            s = $"eq enabled: on\neq precut: {round(peq.max_gain, 1) * 10:.0f}\n"
            for i, filt in enumerate(peq.filters):
                if(i == 0)
                    s += $"eq low shelf filter: {filt.fc:.0f}, {round(filt.q, 1) * 10:.0f}, {round(filt.gain, 1) * 10:.0f}\n"
                elif(i == len(peq.filters) - 1)
                    s += $"eq high shelf filter: {filt.fc:.0f}, {round(filt.q, 1) * 10:.0f}, {round(filt.gain, 1) * 10:.0f}\n"
                else:
                    s += $"eq peak filter {i}: {filt.fc:.0f}, {round(filt.q, 1) * 10:.0f}, {round(filt.gain, 1) * 10:.0f}\n"
            f.write(s)

    @staticmethod
    def _split_path(path):
        """Splits file system path into components."""
        folders = []
        while 1:
            path, folder = os.path.split(path)

            if(folder != "")
                folders.append(folder)
            else:
                if(path != "")
                    folders.append(path)

                break

        folders.reverse()
        return folders

    def minimum_phase_impulse_response(this, fs=DEFAULT_FS, f_res=DEFAULT_F_RES, normalize=true, preamp=DEFAULT_PREAMP):
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
        fr = this.__class__(name="fr_data", frequency=this.Frequency.copy(), raw=this.equalization.copy())
        // Save gain at lowest available frequency
        f_min = np.max([fr.frequency[0], f_res])
        interpolator = InterpolatedUnivariateSpline(np.log10(fr.frequency), fr.raw, k=1)
        gain_f_min = interpolator(np.log10(f_min))
        // Filter length, optimized for FFT speed
        n = round(fs // 2 / f_res)
        n = next_fast_len(n)
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
        ir = firwin2(len(fr.frequency) * 2, fr.frequency, fr.raw, fs=fs)
        // Convert to minimum phase
        ir = minimum_phase(ir, n_fft=len(ir))
        return ir

    def linear_phase_impulse_response(this, fs=DEFAULT_FS, f_res=DEFAULT_F_RES, normalize=true, preamp=DEFAULT_PREAMP):
        """Generates impulse response implementation of equalization filter."""
        // Interpolate to even sample interval
        fr = this.__class__(name="fr_data", frequency=this.Frequency, raw=this.equalization)
        // Save gain at lowest available frequency
        f_min = np.max([fr.frequency[0], f_res])
        interpolator = InterpolatedUnivariateSpline(np.log10(fr.frequency), fr.raw, k=1)
        gain_f_min = interpolator(np.log10(f_min))
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
        ir = firwin2(len(fr.frequency) * 2, fr.frequency, fr.raw, fs=fs)
        return ir

    def write_readme(this, file_path, parametric_peqs=None, fixed_band_peq=None):
        """Writes README.md with picture and Equalizer APO settings."""
        file_path = os.path.abspath(file_path)
        dir_path = os.path.dirname(file_path)
        model = this.name

        // Write model
        s = "# {}\n".format(model)
        s += "See [usage instructions](https://github.com/jaakkopasanen/AutoEq#usage) for more options and info.\n\n"

        // Add parametric EQ settings
        if(parametric_peqs is not None)
            s += "### Parametric EQs\n"
            f = this.generate_frequencies(f_step=DEFAULT_BIQUAD_OPTIMIZATION_F_STEP)
            if(len(parametric_peqs) > 1)
                compound = PEQ(f, parametric_peqs[0].fs)
                n = 0
                filter_ranges = ''
                preamps = ''
                for i, peq in enumerate(parametric_peqs):
                    peq = deepcopy(peq)
                    peq.sort_filters()
                    for filt in peq.filters:
                        compound.add_filter(filt)
                    filter_ranges += $"1-{len(peq.filters) + n}"
                    preamps += $"{-compound.max_gain - 0.1:.1f} dB"
                    if(i < len(parametric_peqs) - 2)
                        filter_ranges += ", "
                        preamps += ", "
                    elif(i == len(parametric_peqs) - 2)
                        filter_ranges += " or "
                        preamps += " or "
                    n += len(peq.filters)
                s += $"You can use filters {filter_ranges}. Apply preamp of {preamps}, respectively.\n\n"
            else:
                compound = PEQ(f, parametric_peqs[0].fs, [])
                for peq in parametric_peqs:
                    peq = deepcopy(peq)
                    peq.sort_filters()
                    for filt in peq.filters:
                        compound.add_filter(filt)
                s += $"Apply preamp of -{compound.max_gain + 0.1:.1f} dB when using parametric equalizer.\n\n"
            s += compound.markdown_table() + "\n\n"

        // Add fixed band eq
        if(fixed_band_peq is not None)
            s += $"### Fixed Band EQs\nWhen using fixed band (also called graphic) equalizer, apply preamp of " \
                 $"**-{fixed_band_peq.max_gain + 0.1:.1f} dB** (if available) and set gains manually with these " \
                 $"parameters.\n\n{fixed_band_peq.markdown_table()}\n\n"

        // Write image link
        img_path = os.path.join(dir_path, model + ".png")
        if(os.path.isfile(img_path))
            img_url = $"./{os.path.split(img_path)[1]}"
            img_url = urllib.parse.quote(img_url, safe="%/:=&?~#+!$,;'@()*[]")
            s += $"### Graphs\n![]({img_url})\n"

        // Write file
        with open(file_path, "w', encoding="utf-8') as f:
            f.write(s)


    def interpolate(this, f=None, f_step=DEFAULT_STEP, pol_order=1, f_min=DEFAULT_F_MIN, f_max=DEFAULT_F_MAX):
        """Interpolates missing values from previous and next value. Resets all but raw data."""
        // Remove None values
        i = 0
        while i < len(this.Raw):
            if(this.Raw[i] is None)
                this.Raw = np.delete(this.Raw, i)
                this.Frequency = np.delete(this.Frequency, i)
            else:
                i += 1

        // Interpolation functions
        keys = "raw error error_smoothed equalization equalized_raw equalized_smoothed target".split()
        interpolators = dict()
        log_f = np.log10(this.Frequency)
        for key in keys:
            if(len(this.__dict__[key]))
                interpolators[key] = InterpolatedUnivariateSpline(log_f, this.__dict__[key], k=pol_order)

        if(f is None)
            this.Frequency = this.generate_frequencies(f_min=f_min, f_max=f_max, f_step=f_step)
        else:
            this.Frequency = np.array(f)

        // Prevent log10 from exploding by replacing zero frequency with small value
        zero_freq_fix = false
        if(this.Frequency[0] == 0)
            this.Frequency[0] = 0.001
            zero_freq_fix = true

        // Run interpolators
        log_f = np.log10(this.Frequency)
        for key in keys:
            if(len(this.__dict__[key]) and key in interpolators)
                this.__dict__[key] = interpolators[key](log_f)

        if(zero_freq_fix)
            // Restore zero frequency
            this.Frequency[0] = 0

        // Everything but the interpolated data is affected by interpolating, reset them
        this.reset(**{key: false for key in keys})

    def center(this, frequency=1000):
        """Removed bias from frequency response.

        Args:
            frequency: Frequency which is set to 0 dB. If this is a list with two values then an average between the two
                       frequencies is set to 0 dB.

        Returns:
            Gain shifted
        """
        equal_energy_fr = this.__class__(name="equal_energy", frequency=this.Frequency.copy(), raw=this.Raw.copy())
        equal_energy_fr.interpolate()
        interpolator = InterpolatedUnivariateSpline(np.log10(equal_energy_fr.frequency), equal_energy_fr.raw, k=1)
        if(type(frequency) in [list, np.ndarray] and len(frequency) > 1)
            // Use the average of the gain values between the given frequencies as the difference to be subtracted
            diff = np.mean(equal_energy_fr.raw[np.logical_and(
                equal_energy_fr.frequency >= frequency[0],
                equal_energy_fr.frequency <= frequency[1]
            )])
        else:
            if(type(frequency) in [list, np.ndarray])
                // List or array with only one element
                frequency = frequency[0]
            // Use the gain value at the given frequency as the difference to be subtracted
            diff = interpolator(np.log10(frequency))

        this.Raw -= diff
        if(len(this.smoothed))
            this.smoothed -= diff
        if(len(this.Error))
            this.Error += diff
        if(len(this.Error_smoothed))
            this.Error_smoothed += diff

        // Everything but raw, smoothed, errors and target is affected by centering, reset them
        this.reset(raw=false, smoothed=false, error=false, error_smoothed=false, target=false)

        return -diff

    def _tilt(this, tilt=DEFAULT_TILT):
        """Creates a tilt for equalization.

        Args:
            tilt: Slope steepness in dB/octave

        Returns:
            Tilted data
        """
        // Center in logarithmic scale
        c = Constants.DEFAULT_F_MIN * np.sqrt(DEFAULT_F_MAX / Constants.DEFAULT_F_MIN)
        // N octaves above center
        n_oct = np.log2(this.Frequency / c)
        return n_oct * tilt

    def create_target(this,
                      bass_boost_gain=DEFAULT_BASS_BOOST_GAIN,
                      bass_boost_fc=DEFAULT_BASS_BOOST_FC,
                      bass_boost_q=DEFAULT_BASS_BOOST_Q,
                      treble_boost_gain=DEFAULT_TREBLE_BOOST_GAIN,
                      treble_boost_fc=DEFAULT_TREBLE_BOOST_FC,
                      treble_boost_q=DEFAULT_TREBLE_BOOST_Q,
                      tilt=None,
                      fs=DEFAULT_FS):
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
        bass_boost = LowShelf(this.Frequency, fs, fc=bass_boost_fc, q=bass_boost_q, gain=bass_boost_gain)
        treble_boost = HighShelf(
            this.Frequency, fs, fc=treble_boost_fc, q=treble_boost_q, gain=treble_boost_gain)
        if(tilt is not None)
            tilt = this._tilt(tilt=tilt)
        else:
            tilt = np.zeros(len(this.Frequency))
        return bass_boost.fr + treble_boost.fr + tilt

    def compensate(this,
                   compensation,
                   bass_boost_gain=DEFAULT_BASS_BOOST_GAIN,
                   bass_boost_fc=DEFAULT_BASS_BOOST_FC,
                   bass_boost_q=DEFAULT_BASS_BOOST_Q,
                   treble_boost_gain=DEFAULT_TREBLE_BOOST_GAIN,
                   treble_boost_fc=DEFAULT_TREBLE_BOOST_FC,
                   treble_boost_q=DEFAULT_TREBLE_BOOST_Q,
                   tilt=None,
                   fs=DEFAULT_FS,
                   sound_signature=None,
                   min_mean_error=false):
        """Sets target and error curves."""
        // Copy and center compensation data
        compensation = this.__class__(name="compensation", frequency=compensation.frequency, raw=compensation.raw)
        compensation.center()

        // Set target
        this.target = compensation.raw + this.create_target(
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
            if(not np.all(sound_signature.frequency == this.Frequency))
                // Interpolate sound signature to match this on the frequency axis
                sound_signature.interpolate(this.Frequency)
            this.target += sound_signature.raw

        // Set error
        this.Error = this.Raw - this.target
        if(min_mean_error)
            // Shift error by it's mean in range 100 Hz to 10 kHz
            delta = np.mean(this.Error[np.logical_and(this.Frequency >= 100, this.Frequency <= 10000)])
            this.Error -= delta
            this.target += delta

        // Smoothed error and equalization results are affected by compensation, reset them
        this.reset(
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

    def _window_size(this, octaves):
        """Calculates moving average window size in indices from octaves."""
        // Octaves to coefficient
        k = 2 ** octaves
        // Calculate average step size in frequencies
        steps = []
        for i in range(1, len(this.Frequency)):
            steps.append(this.Frequency[i] / this.Frequency[i - 1])
        step_size = sum(steps) / len(steps)
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

    def _sigmoid(this, f_lower, f_upper, a_normal=0.0, a_treble=1.0):
        f_center = np.sqrt(f_upper / f_lower) * f_lower
        half_range = np.log10(f_upper) - np.log10(f_center)
        f_center = np.log10(f_center)
        a = expit((np.log10(this.Frequency) - f_center) / (half_range / 4))
        a = a * -(a_normal - a_treble) + a_normal
        return a

    def _smoothen_fractional_octave(this,
                                    data,
                                    window_size=DEFAULT_SMOOTHING_WINDOW_SIZE,
                                    iterations=DEFAULT_SMOOTHING_ITERATIONS,
                                    treble_window_size=None,
                                    treble_iterations=None,
                                    treble_f_lower=DEFAULT_TREBLE_SMOOTHING_F_LOWER,
                                    treble_f_upper=DEFAULT_TREBLE_SMOOTHING_F_UPPER):
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
        if(None in this.Frequency or None in data)
            // Must not contain None values
            throw new ArgumentException("None values present, cannot smoothen!");

        // Normal filter
        y_normal = data
        with warnings.catch_warnings():
            // Savgol filter uses array indexing which is not future proof, ignoring the warning and trusting that this
            // will be fixed in the future release
            warnings.simplefilter("ignore")
            for i in range(iterations):
                y_normal = savgol_filter(y_normal, this._window_size(window_size), 2)

            // Treble filter
            y_treble = data
            for _ in range(treble_iterations):
                y_treble = savgol_filter(y_treble, this._window_size(treble_window_size), 2)

        // Transition weighted with sigmoid
        k_treble = this._sigmoid(treble_f_lower, treble_f_upper)
        k_normal = k_treble * -1 + 1
        return y_normal * k_normal + y_treble * k_treble

    def smoothen_fractional_octave(this,
                                   window_size=DEFAULT_SMOOTHING_WINDOW_SIZE,
                                   iterations=DEFAULT_SMOOTHING_ITERATIONS,
                                   treble_window_size=DEFAULT_TREBLE_SMOOTHING_WINDOW_SIZE,
                                   treble_iterations=DEFAULT_TREBLE_SMOOTHING_ITERATIONS,
                                   treble_f_lower=DEFAULT_TREBLE_SMOOTHING_F_LOWER,
                                   treble_f_upper=DEFAULT_TREBLE_SMOOTHING_F_UPPER):
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
            throw new ArgumentException("Upper transition boundary must be greater than lower boundary");

        // Smoothen raw data
        this.smoothed = this._smoothen_fractional_octave(
            this.Raw,
            window_size=window_size,
            iterations=iterations,
            treble_window_size=treble_window_size,
            treble_iterations=treble_iterations,
            treble_f_lower=treble_f_lower,
            treble_f_upper=treble_f_upper
        )

        if(len(this.Error))
            // Smoothen error data
            this.Error_smoothed = this._smoothen_fractional_octave(
                this.Error,
                window_size=window_size,
                iterations=iterations,
                treble_window_size=treble_window_size,
                treble_iterations=treble_iterations,
                treble_f_lower=treble_f_lower,
                treble_f_upper=treble_f_upper
            )

        // Equalization is affected by smoothing, reset equalization results
        this.reset(
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

    def equalize(this,
                 max_gain=DEFAULT_MAX_GAIN,
                 limit=DEFAULT_MAX_SLOPE,
                 limit_decay=0.0,
                 concha_interference=false,
                 window_size=1 / 12,
                 treble_window_size=2,
                 treble_f_lower=DEFAULT_TREBLE_F_LOWER,
                 treble_f_upper=DEFAULT_TREBLE_F_UPPER,
                 treble_gain_k=DEFAULT_TREBLE_GAIN_K):
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
        fr = FrequencyResponse(name="fr", frequency=this.Frequency, raw=this.Error)
        // Smoothen data heavily in the treble region to avoid problems caused by peakiness
        fr.smoothen_fractional_octave(
            window_size=window_size, treble_window_size=treble_window_size, treble_f_lower=treble_f_lower,
            treble_f_upper=treble_f_upper)

        // Copy data
        x = np.array(fr.frequency)
        y = np.array(-fr.smoothed)  # Inverse of the smoothed error

        // Find peaks and notches
        peak_inds, peak_props = find_peaks(y, prominence=1)
        dip_inds, dip_props = find_peaks(-y, prominence=1)

        if(not len(peak_inds) and not len(dip_inds))
            // No peaks or dips, it's a flat line
            // Use the inverse error as the equalization target
            this.equalization = y
            // Equalized
            this.equalized_raw = this.Raw + this.equalization
            if(len(this.smoothed))
                this.equalized_smoothed = this.smoothed + this.equalization
            return y, fr.smoothed.copy(), np.array([]), np.array([false] * len(y)), np.array([]), \
                np.array([false] * len(y)), np.array([]), np.array([]), len(y) - 1, np.array([false] * len(y))

        else:
            limit_free_mask = this.protection_mask(y, peak_inds, dip_inds)
            if(concha_interference)
                // 8 kHz - 11.5 kHz should not be limit free zone
                limit_free_mask[np.logical_and(x >= 8000, x <= 11500)] = false

            // Find rtl start index
            rtl_start = this.find_rtl_start(y, peak_inds, dip_inds)

            // Find ltr and rtl limitations
            // limited_ltr is y but with slopes limited when traversing left to right
            // clipped_ltr is boolean mask for limited samples when traversing left to right
            // limited_rtl is found using ltr algorithm but with flipped data
            limited_ltr, clipped_ltr, regions_ltr = this.limited_ltr_slope(
                x, y, limit, limit_decay=limit_decay, start_index=0, peak_inds=peak_inds,
                limit_free_mask=limit_free_mask, concha_interference=concha_interference)
            limited_rtl, clipped_rtl, regions_rtl = this.limited_rtl_slope(
                x, y, limit, limit_decay=limit_decay, start_index=rtl_start, peak_inds=peak_inds,
                limit_free_mask=limit_free_mask, concha_interference=concha_interference)

            // ltr and rtl limited curves are combined with min function
            combined = this.__class__(
                name="limiter", frequency=x, raw=np.min(np.vstack([limited_ltr, limited_rtl]), axis=0))

            // Limit treble gain
            gain_k = this._sigmoid(treble_f_lower, treble_f_upper, a_normal=1.0, a_treble=treble_gain_k)
            combined.raw *= gain_k

            // Gain can be reduced in the treble region
            // Clip positive gain to max gain
            combined.raw = np.min(np.vstack([combined.raw, np.ones(combined.raw.shape) * max_gain]), axis=0)
            // Smoothen the curve to get rid of hard kinks
            combined.smoothen_fractional_octave(window_size=1 / 5, treble_window_size=1 / 5)

            // Equalization curve
            this.equalization = combined.smoothed

        // Equalized
        this.equalized_raw = this.Raw + this.equalization
        if(len(this.smoothed))
            this.equalized_smoothed = this.smoothed + this.equalization

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
        if(len(peak_inds) and (not len(dip_inds) or peak_inds[-1] > dip_inds[-1]))
            // Last peak is after last dip, add new dip after the last peak at the minimum
            last_dip_ind = np.argmin(y[peak_inds[-1]:]) + peak_inds[-1]
            dip_inds = np.concatenate([dip_inds, [last_dip_ind]])
            dip_levels = y[dip_inds]
        else:
            dip_inds = np.concatenate([dip_inds, [-1]])
            dip_levels = y[dip_inds]
            dip_levels[-1] = np.min(y)

        mask = np.zeros(len(y)).astype(bool)
        if(len(dip_inds) < 3)
            return mask

        for i in range(1, len(dip_inds) - 1):
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
        start_index = len(x) - start_index - 1
        if(peak_inds is not None)
            peak_inds = len(x) - peak_inds - 1
        if(limit_free_mask is not None)
            limit_free_mask = np.flip(limit_free_mask)
        limited_rtl, clipped_rtl, regions_rtl = cls.limited_ltr_slope(
            x, np.flip(y), limit, limit_decay=limit_decay, start_index=start_index, peak_inds=peak_inds,
            limit_free_mask=limit_free_mask, concha_interference=concha_interference)
        limited_rtl = np.flip(limited_rtl)
        clipped_rtl = np.flip(clipped_rtl)
        regions_rtl = len(x) - regions_rtl - 1
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
        for i in range(len(x)):
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
                local_limit *= (1 - limit_decay) ** np.log2(x[i] / x[regions[-1][0]])

            if(slope > local_limit and (limit_free_mask is None or not limit_free_mask[i]))
                // Slope between the two samples is greater than the local maximum slope, clip to the max
                if(not clipped[-1])
                    // Start of clipped region
                    regions.append([i])
                clipped.append(true)
                // Add value with limited change
                octaves = np.log(x[i] / x[i - 1]) / np.log(2)
                limited.append(limited[-1] + local_limit * octaves)

            else:
                // Moderate slope, no need to limit
                limited.append(y[i])

                if(clipped[-1])
                    // Previous sample clipped but this one didn"t, means it"s the end of clipped region
                    // Add end index to the region
                    regions[-1].append(i + 1)

                    region_start = regions[-1][0]
                    if(peak_inds is not None and not np.any(np.logical_and(peak_inds >= region_start, peak_inds < i)))
                        // None of the peak indices found in the current region, discard limitations
                        limited[region_start:i] = y[region_start:i]
                        clipped[region_start:i] = [false] * (i - region_start)
                        regions.pop()
                clipped.append(false)

        if(len(regions) and len(regions[-1]) == 1)
            regions[-1].append(len(x) - 1)

        return np.array(limited), np.array(clipped), np.array(regions)

    @staticmethod
    def log_log_gradient(f0, f1, g0, g1):
        """Calculates gradient (derivative) in dB per octave."""
        octaves = np.log(f1 / f0) / np.log(2)
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
        if(len(peak_inds) and (not len(dip_inds) or peak_inds[-1] > dip_inds[-1]))
            // Last peak is a positive peak
            if(len(dip_inds))
                // Find index on the right side of the peak where the curve crosses the last dip level
                rtl_start = np.argwhere(y[peak_inds[-1]:] <= y[dip_inds[-1]])
            else:
                // There are no dips, use the minimum of the first and the last value of y
                rtl_start = np.argwhere(y[peak_inds[-1]:] <= max(y[0], y[-1]))
            if(len(rtl_start))
                rtl_start = rtl_start[0, 0] + peak_inds[-1]
            else:
                rtl_start = len(y) - 1
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
    def init_plot(fig=None, ax=None, f_min=DEFAULT_F_MIN, f_max=DEFAULT_F_MAX, a_min=None, a_max=None, ):
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

    def plot_graph(this,
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
                   f_min=DEFAULT_F_MIN,
                   f_max=DEFAULT_F_MAX,
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
        if(not len(this.Frequency))
            throw new ArgumentException("\'frequency\" has no data!');

        fig, ax = this.__class__.init_plot(fig=fig, ax=ax, f_min=f_min, f_max=f_max, a_min=a_min, a_max=a_max)

        if(target and len(this.target))
            ax.plot(
                this.Frequency, this.target,
                **this.kwarg_defaults(target_plot_kwargs, label="Target", linewidth=5, color="lightblue")
            )

        if(smoothed and len(this.smoothed))
            ax.plot(
                this.Frequency, this.smoothed,
                **this.kwarg_defaults(smoothed_plot_kwargs, label="Raw Smoothed", linewidth=5, color="lightgrey")
            )

        if(error_smoothed and len(this.Error_smoothed))
            ax.plot(
                this.Frequency, this.Error_smoothed,
                **this.kwarg_defaults(error_smoothed_plot_kwargs, label="Error Smoothed", linewidth=5, color="pink")
            )

        if(raw and len(this.Raw))
            ax.plot(
                this.Frequency, this.Raw,
                **this.kwarg_defaults(raw_plot_kwargs, label="Raw", linewidth=1, color=color)
            )

        if(error and len(this.Error))
            ax.plot(
                this.Frequency, this.Error,
                **this.kwarg_defaults(error_plot_kwargs, label="Erro@", linewidth=1, color="red")
            )

        if(equalization and len(this.equalization))
            ax.plot(
                this.Frequency, this.equalization,
                **this.kwarg_defaults(equalization_plot_kwargs, label="Equalization", linewidth=5, color="lightgreen")
            )

        if(parametric_eq and len(this.parametric_eq))
            ax.plot(
                this.Frequency, this.parametric_eq,
                **this.kwarg_defaults(parametric_eq_plot_kwargs, label="Parametric Eq", linewidth=1, color="darkgreen")
            )

        if(fixed_band_eq and len(this.fixed_band_eq))
            ax.plot(
                this.Frequency, this.fixed_band_eq,
                **this.kwarg_defaults(
                    fixed_band_eq_plot_kwargs,
                    label="Fixed Band Eq", linewidth=1, color="darkgreen", linestyle="--"
                )
            )

        if(equalized and len(this.equalized_raw))
            ax.plot(
                this.Frequency, this.equalized_raw,
                **this.kwarg_defaults(equalized_plot_kwargs, label="Equalized", linewidth=1, color="blue")
            )

        ax.set_title(this.name)
        if(len(ax.lines) > 0)
            ax.legend(fontsize=8)

        if(file_path is not None)
            file_path = os.path.abspath(file_path)
            fig.savefig(file_path, dpi=120)
            im = Image.open(file_path)
            im = im.convert('P', palette=Image.ADAPTIVE, colors=60)
            im.save(file_path, optimize=true)
        if(show)
            plt.show()
        elif(close)
            plt.close(fig)
        return fig, ax

    def harman_onear_preference_score(this):
        """Calculates Harman preference score for over-ear and on-ear headphones.

        Returns:
            - score: Preference score
            - std: Standard deviation of error
            - slope: Slope of linear regression of error
        """
        fr = this.copy()
        fr.interpolate(HARMAN_ONEAR_PREFERENCE_FREQUENCIES)
        sl = np.logical_and(fr.frequency >= 50, fr.frequency <= 10000)
        x = fr.frequency[sl]
        y = fr.error[sl]

        std = np.std(y, ddof=1)  # ddof=1 is required to get the exact same numbers as the Excel from Listen Inc gives
        slope, _, _, _, _ = linregress(np.log(x), y)
        score = 114.490443008238 - 12.62 * std - 15.5163857197367 * np.abs(slope)

        return score, std, slope

    def harman_inear_preference_score(this):
        """Calculates Harman preference score for in-ear headphones.

        Returns:
            - score: Preference score
            - std: Standard deviation of error
            - slope: Slope of linear regression of error
            - mean: Mean of absolute error
        """
        fr = this.copy()
        fr.interpolate(HARMAN_INEAR_PREFENCE_FREQUENCIES)
        sl = np.logical_and(fr.frequency >= 20, fr.frequency <= 10000)
        x = fr.frequency[sl]
        y = fr.error[sl]

        std = np.std(y, ddof=1)  # ddof=1 is required to get the exact same numbers as the Excel from Listen Inc gives
        slope, _, _, _, _ = linregress(np.log(x), y)
        // Mean of absolute of error centered by 500 Hz
        delta = fr.error[np.where(fr.frequency == 500.0)[0][0]]
        y = fr.error[np.logical_and(fr.frequency >= 40, fr.frequency <= 10000)] - delta
        mean = np.mean(np.abs(y))
        // Final score
        score = 100.0795 - 8.5 * std - 6.796 * np.abs(slope) - 3.475 * mean

        return score, std, slope, mean

    def process(this,
                compensation=None,
                min_mean_error=false,
                bass_boost_gain=None,
                bass_boost_fc=None,
                bass_boost_q=None,
                treble_boost_gain=None,
                treble_boost_fc=None,
                treble_boost_q=None,
                tilt=None,
                fs=DEFAULT_FS,
                sound_signature=None,
                max_gain=DEFAULT_MAX_GAIN,
                concha_interference=false,
                window_size=DEFAULT_SMOOTHING_WINDOW_SIZE,
                treble_window_size=DEFAULT_TREBLE_SMOOTHING_WINDOW_SIZE,
                treble_f_lower=DEFAULT_TREBLE_F_LOWER,
                treble_f_upper=DEFAULT_TREBLE_F_UPPER,
                treble_gain_k=DEFAULT_TREBLE_GAIN_K):
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
        this.interpolate()
        this.center()
        this.compensate(
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
        this.smoothen_fractional_octave(
            window_size=window_size,
            treble_window_size=treble_window_size,
            treble_f_lower=treble_f_lower,
            treble_f_upper=treble_f_upper
        )
        this.equalize(
            max_gain=max_gain, concha_interference=concha_interference, treble_f_lower=treble_f_lower,
            treble_f_upper=treble_f_upper, treble_gain_k=treble_gain_k)
        */
}