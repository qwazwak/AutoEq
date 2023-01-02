using AutoEQ.Core;
using CommandLine;

namespace AutoEQ.ConsoleUI;
/*
from autoeq.constants import DEFAULT_MAX_GAIN, DEFAULT_TREBLE_F_LOWER, DEFAULT_TREBLE_F_UPPER, \
    DEFAULT_TREBLE_GAIN_K, DEFAULT_FS, DEFAULT_BIT_DEPTH, DEFAULT_PHASE, DEFAULT_F_RES, DEFAULT_BASS_BOOST_FC, \
    DEFAULT_BASS_BOOST_Q, DEFAULT_SMOOTHING_WINDOW_SIZE, DEFAULT_TREBLE_SMOOTHING_WINDOW_SIZE, DEFAULT_TREBLE_BOOST_FC, \
    DEFAULT_TREBLE_BOOST_Q, DEFAULT_PREAMP
from autoeq.batch_processing import batch_processing
 */


public class AutoEQ2Options : IAutoEQ2Options
{
    private static readonly Lazy<AutoEQ2Options> _Instance = new(() => new());
    public static AutoEQ2Options Instance => _Instance.Value;
    private AutoEQ2Options() { }
    [Option('v', "verbose", Required = false, Default = false, HelpText = "Enables verbose (extra) logging")]
    public bool Verbose { get; set; } = false;

    [Option('i', "input-dir", Required = true, HelpText = "Path to input data directory. Will look for CSV files in the data directory and recursively in sub-directories.")]
    public string InputDirectory { get; set; } = null!;

    [Option('o', "output-dir", Required = true, HelpText = "Path to results directory. Will keep the same relative paths for files found in input-dir.")]
    public string OutputDirectory { get; set; } = null!;

    [Option('s', "standardize-input", Required = false, Default = false, HelpText = "Overwrite input data in standardized sampling and bias?")]
    public bool StandardizeInput { get; set; } = false;

    [Option('n', "new-only", Required = false, Default = false, HelpText = "Only process input files which don\"t have results in output directory.")]
    public bool NewOnly { get; set; } = false;

    [Option('c', "compensation", Required = false, Default = false, HelpText = "File path to CSV containing compensation (target) curve. Compensation is necessary when equalizing because all input data is raw microphone data. See \"compensation\", \"innerfidelity / resources\" and \"headphonecom / resources\".")]
    public bool Compensation { get; set; } = false;

    [Option('e', "equalize", Required = false, Default = false, HelpText = "Will run equalization if this parameter exists, no value needed.")]
    public bool Equalize { get => true; set => ConsoleUI.WriteLine("\"equalize\" parameter is no longer supported. The equalization target is created automatically every time."); }


    /*
     
    if "parametric_eq_config" in args:
            if not os.path.isfile(args["parametric_eq_config"]):
            // Named configurations, split by commas
            args["parametric_eq_config"] = args["parametric_eq_config"].split(',')

     */
    private string? _ParametricEQConfig = null;
    private ICollection<string>? _ParametricEQConfigs = null;
    public ICollection<string> ParametricEQConfigs { get => _ParametricEQConfigs ?? ; set => _ParametricEQConfigs = value != null && value.Count > 0 ? value : null; }
    [Option('p', "parametric-eq", Required = false, Default = null, HelpText = "Will run equalization if this parameter exists, no value needed.")]
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
    }     /*    arg_parser.add_argument("--parametric-eq", action="store_true",
                            help="Will produce parametric eq settings if this parameter exists, no value needed.")
    arg_parser.add_argument("--fixed-band-eq", action="store_true",
                            help="Will produce fixed band eq settings if this parameter exists, no value needed.")
    arg_parser.add_argument("--rockbox", action="store_true",
                            help="Will produce a Rockbox .cfg file with 10 band eq settings if this parameter exists,"
                                 "no value needed.")
    arg_parser.add_argument("--ten-band-eq", action="store_true",
                            help="Shortcut parameter for activating standard ten band eq optimization.")
    arg_parser.add_argument("--parametric-eq-config", type=str,
                            default="4_PEAKING_WITH_LOW_SHELF,4_PEAKING_WITH_HIGH_SHELF",
                            help="Name of parametric equalizer configuration or a path to a configuration file. "
                                 "Available named configurations are "10_PEAKING" for 10 peaking filters, "
                                 ""8_PEAKING_WITH_SHELVES" for 8 peaking filters and a low shelf at 105 Hz for bass "
                                 "adjustment and a high shelf at 10 kHz for treble adjustment, "
                                 ""4_PEAKING_WITH_LOW_SHELF" for 4 peaking filters and a low shelf at 105 Hz for bass "
                                 "adjustment, "4_PEAKING_WITH_HIGH_SHELF" for 4 peaking filters and a high shelf "
                                 "at 10 kHz for treble adjustments. You can give multiple named configurations by "
                                 "separating the names with commas and filter sets will be built on top of each other. "
                                 "When the value is a file path, the file will be read and used as a configuration. "
                                 "The file needs to be a YAML file with "filters" field as a list of filter "
                                 "configurations, each of which can define "fc", "min_fc", "max_fc", "q", "min_q", "
                                 ""max_q", "gain", "min_gain", "max_gain" and "type" fields. When the fc, q or gain "
                                 "value is given, the parameter won\"t be optimized for the filter. "type" needs to '
                                 "be either "LOW_SHELF", "PEAKING" or "HIGH_SHELF". Also "filter_defaults" field is "
                                 "supported on the top level and it can have the same fields as the filters do. "
                                 "All fields missing from the filters will be read from "filter_defaults". "
                                 "Defaults to "4_PEAKING_WITH_LOW_SHELF,4_PEAKING_WITH_HIGH_SHELF". "
                                 "Optimizer behavior can be adjusted by defining "optimizer" field which has fields "
                                 ""min_f" and "max_f" for lower and upper bounds of the optimization range, "max_time" "
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
                                 f"Defaults to {DEFAULT_FS}.")
    arg_parser.add_argument("--bit-depth", type=int, default=DEFAULT_BIT_DEPTH,
                            help="Number of bits for every sample in impulse response. "
                                 f"Defaults to {DEFAULT_BIT_DEPTH}.")
    arg_parser.add_argument("--phase", type=str, default=DEFAULT_PHASE,
                            help="Impulse response phase characteristic. "minimum", "linear" or "both". "
                                 f"Defaults to "{DEFAULT_PHASE}"")
    arg_parser.add_argument("--f-res", type = float, default = DEFAULT_F_RES,
                            help = "Frequency resolution for impulse responses. If this is 20 then impulse response "
                                 "frequency domain will be sampled every 20 Hz. Filter length for "
                                 f"impulse responses will be fs/f_res. Defaults to {DEFAULT_F_RES}.")
    */
    private string? _BassBoost = null;
    private const string _BassBoostAttributeString = $"Bass boost shelf. Sub-bass frequencies will be boosted by this amount. Can be either a single value for a gain in dB or a comma separated list of three values for parameters of a low shelf filter, where the first is gain in dB, second is center frequency (Fc) in Hz and the last is quality (Q). When only a single value (gain) is given, default values for Fc and Q are used which are {Constants.DEFAULT_BASS_BOOST_FC} Hz and {Constants.DEFAULT_BASS_BOOST_Q}, respectively. For example \"--bass-boost = 6\" or \"--bass-boost = 9.5, 150, 0.69\".";
    [Option('b', "bass-boost", Required = false, Default = null, HelpText = _BassBoostAttributeString)]
    public string? BassBoost
    {
        get => _BassBoost;
        set => ApplyBoost(value, "bass-boost", Constants.DEFAULT_BASS_BOOST_FC, Constants.DEFAULT_BASS_BOOST_Q, ref _bass_boost_gain, ref _bass_boost_fc, ref _bass_boost_q);
    }
    private double? _bass_boost_gain;
    private double? _bass_boost_fc;
    private double? _bass_boost_q;

    public double? bass_boost_gain { get => _bass_boost_gain; set => _bass_boost_gain = value; }
    public double? bass_boost_fc { get => _bass_boost_fc; set => _bass_boost_fc = value; }
    public double? bass_boost_q { get => _bass_boost_q; set => _bass_boost_q = value; }

    private string? _TrebleBoost = null;
    private const string _TrebleBoostAttributeString = $"Treble boost shelf. Sub-treble frequencies will be boosted by this amount. Can be either a single value for a gain in dB or a comma separated list of three values for parameters of a low shelf filter, where the first is gain in dB, second is center frequency (Fc) in Hz and the last is quality (Q). When only a single value (gain) is given, default values for Fc and Q are used which are {Constants.DEFAULT_BASS_BOOST_FC} Hz and {Constants.DEFAULT_BASS_BOOST_Q}, respectively. For example \"--treble-boost = 6\" or \"--treble-boost = 9.5, 150, 0.69\".";
    [Option('b', "treble-boost", Required = false, Default = null, HelpText = _TrebleBoostAttributeString)]
    public string? TrebleBoost
    {
        get => _TrebleBoost; set => ApplyBoost(value, "treble-boost", Constants.DEFAULT_TREBLE_BOOST_FC, Constants.DEFAULT_TREBLE_BOOST_Q, ref TrebleBoosts, ref _treble_boost_gain, ref _treble_boost_fc, ref _treble_boost_q);

    }
    private double? _treble_boost_gain;
    private double? _treble_boost_fc;
    private double? _treble_boost_q;

    public double? treble_boost_gain { get => _treble_boost_gain; set => _treble_boost_gain = value; }
    public double? treble_boost_fc { get => _treble_boost_fc; set => _treble_boost_fc = value; }
    public double? treble_boost_q { get => _treble_boost_q; set => _treble_boost_q = value; }

    private static void ApplyBoost(string? Input, string Argument, double Default_FC_Boost, double Default_Q_Boost, ref double? gain, ref double? fc, ref double? q)
    {
        string[]? Boosts = Input?.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (Boosts != null)
        {
            if (Boosts.Length == 1)
            {
                gain = double.Parse(Boosts[0]);
                fc = Default_FC_Boost;
                q = Default_Q_Boost;
            }
            else if (Boosts.Length == 3)
            {
                gain = double.Parse(Boosts[0]);
                fc = double.Parse(Boosts[1]);
                q = double.Parse(Boosts[2]);
            }
            else
                throw new ArgumentOutOfRangeException(Argument, $"\"--{Argument}\" must have one value or three values separated by commas!");
        }
    }
    /*
    arg_parser.add_argument("--treble-boost", type = str, default = argparse.SUPPRESS,
                            help = "Treble boost shelf. > 10 kHz frequencies will be boosted by this amount. Can be "
                                 "either a single value for a gain in dB or a comma separated list of three values "
                                 "for parameters of a high shelf filter, where the first is gain in dB, second is "
                                 "center frequency (Fc) in Hz and the last is quality (Q). When only a single "
                                 "value (gain) is given, default values for Fc and Q are used which are "
                                 f"{DEFAULT_TREBLE_BOOST_FC} Hz and {DEFAULT_TREBLE_BOOST_Q}, "
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
    arg_parser.add_argument("--max-gain", type = float, default = DEFAULT_MAX_GAIN,
                            help = "Maximum positive gain in equalization. Higher max gain allows to equalize deeper "
                                 "dips in  frequency response but will limit output volume if no analog gain is "
                                 "available because positive gain requires negative digital preamp equal to "
                                 f"maximum positive gain. Defaults to {DEFAULT_MAX_GAIN}.")
    arg_parser.add_argument("--window-size", type = float, default = DEFAULT_SMOOTHING_WINDOW_SIZE,
                            help = "Smoothing window size in octaves.")
    arg_parser.add_argument("--treble-window-size", type = float, default = DEFAULT_TREBLE_SMOOTHING_WINDOW_SIZE,
                            help = "Smoothing window size in octaves in the treble region.")
    arg_parser.add_argument("--treble-f-lower", type = float, default = DEFAULT_TREBLE_F_LOWER,
                            help = "Lower bound for transition region between normal and treble frequencies. Treble "
                                 "frequencies can have different max gain and gain K. Defaults to "
                                 f"{DEFAULT_TREBLE_F_LOWER}.")
    arg_parser.add_argument("--treble-f-upper", type = float, default = DEFAULT_TREBLE_F_UPPER,
                            help = "Upper bound for transition region between normal and treble frequencies. Treble "
                                 "frequencies can have different max gain and gain K. Defaults to "
                                 f"{DEFAULT_TREBLE_F_UPPER}.")
    arg_parser.add_argument("--treble-gain-k", type = float, default = DEFAULT_TREBLE_GAIN_K,
                            help = "Coefficient for treble gain, affects both positive and negative gain. Useful for "
                                 "disabling or reducing equalization power in treble region. Defaults to "
                                 f"{DEFAULT_TREBLE_GAIN_K}.")
*/
    [Option('t', "thread-count", Required = false, Default = 1, HelpText = "Amount of threads to use for processing results. If set to \"max\" all the threads available will be used. Using more threads result in higher memory usage. Defaults to 1.")]
    public string ThreadCountStr
    {
        get => ThreadCount.ToString();
        set
        {
            string input = value.Trim();
            if (input.Equals("max", StringComparison.OrdinalIgnoreCase))
                ThreadCount = Environment.ProcessorCount;
            else if (int.TryParse(value, out int Count))
            {
                if(Count <= 0 || Count > Environment.ProcessorCount)
                    throw new ArgumentOutOfRangeException($"thread-count must be atleast 0 and less than {Environment.ProcessorCount}", "thread-count");
                ThreadCount = Count;
            }
            else
                throw new ArgumentException($"Could not parse {value} to a number", "thread-count");
        }
    }
    public int ThreadCount { get; set; }
    [Option('a', "preamp", Required = false, Default = Constants.DEFAULT_PREAMP, HelpText = "Extra pre-amplification to be applied to equalizer settings in dB")]
    public double PreAmp { get; set; }
}
public class Class1
{
    public static void Main(string[] Argv)
    {
        ParseArguments(Argv);
        BatchProcessing.batch_processing(batch_processing);
    }
    public static void ParseArguments(string[] Argv)
    {
        Parser.Default.ParseArguments<AutoEQ2Options>(() => AutoEQ2Options.Instance, Argv)
               .WithParsed<AutoEQ2Options>(args =>
               {

                   if (args.Verbose)
                   {
                       ConsoleUI.WriteLine($"Verbose output enabled. Current Arguments: -v {args.Verbose}");
                       ConsoleUI.WriteLine("Quick Start Example! App is in Verbose mode!");
                   }

                   // Replace hyphens with underscores to be compatible with the batch_processing method signature
                   //args = {key.replace('-', '_'): val for key, val in args.items()}
                   if (args.Equalize)
                   {
                       ConsoleUI.WriteLine("\"equalize\" parameter is no longer supported. The equalization target is created automatically every time.");
                   }


                   //   if "fs" in args and args["fs"] is not None:
                   //        args["fs"] = [int(x) for x in args["fs"].split(',')]

               });
    }
}