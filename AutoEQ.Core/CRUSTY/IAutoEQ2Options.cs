namespace AutoEQ2.Core;

public interface IAutoEQ2Options
{
    public bool Verbose { get; }

    public string InputDirectory { get; }

    public string OutputDirectory { get; }

    public bool StandardizeInput { get; }

    public bool NewOnly { get; }

    public bool Compensation { get; }
    /*
    private string? _ParametricEQConfig = null;
    private ICollection<string>? _ParametricEQConfigs = null;
    public ICollection<string> ParametricEQConfigs { get => _ParametricEQConfigs ?? ; set => _ParametricEQConfigs = value != null && value.Count > 0 ? value : null; }
    public string? ParametricEQConfig { get => _ParametricEQConfig; set
        {
            ParametricEQConfig = value;
            if (ParametricEQConfig == null)
                return;
            if (string.IsNullOrWhiteSpace(ParametricEQConfig))
                throw new ArgumentException("Must not be whitespace", nameof(ParametricEQConfig));
            {
                if (!File.Exists(ParametricEQConfig))
                {
                    // Named configurations, split by commas
                    args["parametric_eq_config"] = args["parametric_eq_config"].split(',');
                }
            }
        }
    }
    */
    /*    arg_parser.add_argument("--parametric-eq", action="store_true",
                            help="Will produce parametric eq settings if this parameter exists, no value needed.")
    arg_parser.add_argument("--fixed-band-eq", action="store_true",
                            help="Will produce fixed band eq settings if this parameter exists, no value needed.")
    arg_parser.add_argument("--rockbox", action="store_true",
                            help="Will produce a Rockbox .cfg file with 10 band eq settings if this parameter exists,"
                                 "no value needed.")
    arg_parser.add_argument("--ten-band-eq", action="store_true",
                            help="Shortcut parameter for activating standard ten band eq optimization.")
    arg_parser.add_argument("--parametric-eq-config", type=str,
                            default="4_PEAKING_WITH_LOW_SHELF,4_PEAKING_WITH_HIGH_SHEL$",
                            help="Name of parametric equalizer configuration or a path to a configuration file. "
                                 "Available named configurations are "10_PEAKING" for 10 peaking filters, "
                                 ""8_PEAKING_WITH_SHELVES" for 8 peaking filters and a low shelf at 105 Hz for bass "
                                 "adjustment and a high shelf at 10 kHz for treble adjustment, "
                                 ""4_PEAKING_WITH_LOW_SHEL$" for 4 peaking filters and a low shelf at 105 Hz for bass "
                                 "adjustment, "4_PEAKING_WITH_HIGH_SHEL$" for 4 peaking filters and a high shelf "
                                 "at 10 kHz for treble adjustments. You can give multiple named configurations by "
                                 "separating the names with commas and filter sets will be built on top of each other. "
                                 "When the value is a file path, the file will be read and used as a configuration. "
                                 "The file needs to be a YAML file with "filters" field as a list of filter "
                                 "configurations, each of which can define "fc", "min_fc", "max_fc", "q", "min_q", "
                                 ""max_q", "gain", "min_gain", "max_gain" and "type" fields. When the fc, q or gain "
                                 "value is given, the parameter won\"t be optimized for the filter. "type" needs to '
                                 "be either "LOW_SHEL$", "PEAKING" or "HIGH_SHEL$". Also "filter_defaults" field is "
                                 "supported on the top level and it can have the same fields as the filters do. "
                                 "All fields missing from the filters will be read from "filter_defaults". "
                                 "Defaults to "4_PEAKING_WITH_LOW_SHELF,4_PEAKING_WITH_HIGH_SHEL$". "
                                 "Optimizer behavior can be adjusted by defining "optimizer" field which has fields "
                                 ""min_$" and "max_$" for lower and upper bounds of the optimization range, "max_time" "
                                 "for maximum optimization duration in seconds, "target_loss" for RMSE target level "
                                 "upon reaching which the optimization is ended, "min_change_rate" for minimum rate "
                                 "of improvement in db/s and "min_std" for minimum standard deviation of the last few "
                                 "loss values. "min_change_rate" and "min_std" end the optimization when further time "
                                 "spent optimizing can\"t be expected to improve the results dramatically. See '
                                 "peq.yaml for an example."),
    arg_parser.add_argument("--fixed-band-eq-config", type=str, default="10_BAND_GRAPHIC_EQ",
                            help="Path to fixed band equalizer configuration. The file format is the same YAML as "
                                 "for parametric equalizer.")
    arg_parser.add_argument("--convolution-eq", action="store_true",
                            help="Will produce impulse response for convolution equalizers if this parameter exists, "
                                 "no value needed.")
    arg_parser.add_argument("--fs", type=str, default=str(DEFAULT_FS),
                            help="Sampling frequency in Hertz for impulse response and parametric eq filters. Single "
                                 "value or multiple values separated by commas eg 44100,48000. When multiple values "
                                 "are given only the first one will be used for parametric eq. "
                                 $"Defaults to {DEFAULT_FS}.")
    arg_parser.add_argument("--bit-depth", type=int, default=DEFAULT_BIT_DEPTH,
                            help="Number of bits for every sample in impulse response. "
                                 $"Defaults to {DEFAULT_BIT_DEPTH}.")
    arg_parser.add_argument("--phase", type=str, default=DEFAULT_PHASE,
                            help="Impulse response phase characteristic. "minimum", "linear" or "both". "
                                 $"Defaults to "{DEFAULT_PHASE}"")
    arg_parser.add_argument("--f-res", type = float, default = Constants.DEFAULT_F_RES,
                            help = "Frequency resolution for impulse responses. If this is 20 then impulse response "
                                 "frequency domain will be sampled every 20 Hz. Filter length for "
                                 $"impulse responses will be fs/f_res. Defaults to {DEFAULT_F_RES}.")
    */
    /*
    private string? _BassBoost = null;
    private const string _BassBoostAttributeString = $"Bass boost shelf. Sub-bass frequencies will be boosted by this amount. Can be either a single value for a gain in dB or a comma separated list of three values for parameters of a low shelf filter, where the first is gain in dB, second is center frequency (Fc) in Hz and the last is quality (Q). When only a single value (gain) is given, default values for Fc and Q are used which are {Constants.DEFAULT_BASS_BOOST_FC} Hz and {Constants.DEFAULT_BASS_BOOST_Q}, respectively. For example \"--bass-boost = 6\" or \"--bass-boost = 9.5, 150, 0.69\".";
    public string? BassBoost
    {
        get => _BassBoost;
        set => ApplyBoost(value, "bass-boost", Constants.DEFAULT_BASS_BOOST_FC, Constants.DEFAULT_BASS_BOOST_Q, ref _bass_boost_gain, ref _bass_boost_fc, ref _bass_boost_q);
    }
    private double? _bass_boost_gain;
    private double? _bass_boost_fc;
    private double? _bass_boost_q;
    */
    public double? bass_boost_gain { get; }
    public double? bass_boost_fc { get; }
    public double? bass_boost_q { get; }
    public double? treble_boost_gain { get; }
    public double? treble_boost_fc { get; }
    public double? treble_boost_q { get; }
    /*
    arg_parser.add_argument("--treble-boost", type = str, default = argparse.SUPPRESS,
                            help = "Treble boost shelf. > 10 kHz frequencies will be boosted by this amount. Can be "
                                 "either a single value for a gain in dB or a comma separated list of three values "
                                 "for parameters of a high shelf filter, where the first is gain in dB, second is "
                                 "center frequency (Fc) in Hz and the last is quality (Q). When only a single "
                                 "value (gain) is given, default values for Fc and Q are used which are "
                                 $"{DEFAULT_TREBLE_BOOST_FC} Hz and {DEFAULT_TREBLE_BOOST_Q}, "
                                 "respectively. For example "--treble - boost = 3" or "--treble - boost = -4, 12000, 0.69".")
    arg_parser.add_argument("--tilt", type = float, default = argparse.SUPPRESS,
                            help = "Target tilt in dB/octave. Positive value (upwards slope) will result in brighter "
                                 "frequency response and negative value (downwards slope) will result in darker "
                                 "frequency response. 1 dB/octave will produce nearly 10 dB difference in "
                                 "desired value between 20 Hz and 20 kHz. Tilt is applied with bass boost and both "
                                 "will affect the bass gain.")
    arg_parser.add_argument("--sound-signature", type = str,
                            help = "File path to a sound signature CSV file. Sound signature is added to the "
                                 "compensation curve. Error data will be used as the sound signature target if "
                                 "the CSV file contains an error column and otherwise the raw column will be used. "
                                 "This means there are two different options for using sound signature: 1st is "
                                 "pointing it to a result CSV file of a previous run and the 2nd is to create a "
                                 "CSV file with just frequency and raw columns by hand (or other means). The Sound "
                                 "signature graph will be interpolated so any number of point at any frequencies "
                                 "will do, making it easy to create simple signatures with as little as two or "
                                 "three points.")
    arg_parser.add_argument("--max-gain", type = float, default = Constants.DEFAULT_MAX_GAIN,
                            help = "Maximum positive gain in equalization. Higher max gain allows to equalize deeper "
                                 "dips in  frequency response but will limit output volume if no analog gain is "
                                 "available because positive gain requires negative digital preamp equal to "
                                 $"maximum positive gain. Defaults to {DEFAULT_MAX_GAIN}.")
    arg_parser.add_argument("--window-size", type = float, default = Constants.DEFAULT_SMOOTHING_WINDOW_SIZE,
                            help = "Smoothing window size in octaves.")
    arg_parser.add_argument("--treble-window-size", type = float, default = Constants.DEFAULT_TREBLE_SMOOTHING_WINDOW_SIZE,
                            help = "Smoothing window size in octaves in the treble region.")
    arg_parser.add_argument("--treble-f-lower", type = float, default = Constants.DEFAULT_TREBLE_F_LOWER,
                            help = "Lower bound for transition region between normal and treble frequencies. Treble "
                                 "frequencies can have different max gain and gain K. Defaults to "
                                 $"{DEFAULT_TREBLE_F_LOWER}.")
    arg_parser.add_argument("--treble-f-upper", type = float, default = Constants.DEFAULT_TREBLE_F_UPPER,
                            help = "Upper bound for transition region between normal and treble frequencies. Treble "
                                 "frequencies can have different max gain and gain K. Defaults to "
                                 $"{DEFAULT_TREBLE_F_UPPER}.")
    arg_parser.add_argument("--treble-gain-k", type = float, default = Constants.DEFAULT_TREBLE_GAIN_K,
                            help = "Coefficient for treble gain, affects both positive and negative gain. Useful for "
                                 "disabling or reducing equalization power in treble region. Defaults to "
                                 $"{DEFAULT_TREBLE_GAIN_K}.")
*/
    public int ThreadCount { get; }
    public double PreAmp { get; }
}
