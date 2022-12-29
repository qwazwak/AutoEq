using System;
using System.Collections.Generic;
using System.IO;

namespace AutoEQ.Core;
public class BatchProcessing
{
    public dynamic PEQ_CONFIGS { get; set; }
    public string input_dir { get; set; }
    public string output_dir { get; set; }
    public bool new_only { get; set; }
    public bool standardize_input { get; set; } = false;
    public bool compensation { get; set; }
    public bool parametric_eq { get; set; } = false;
    public bool fixed_band_eq { get; set; } = false;
    public bool rockbox { get; set; } = false;
    public bool ten_band_eq { get; set; } = false;
    public dynamic parametric_eq_config { get; set; }
    public dynamic fixed_band_eq_config { get; set; }
    public bool convolution_eq { get; set; } = false;
    public IList<int> fs { get; set; } = new List<int>(Constants.DEFAULT_FS);
    public int bit_depth { get; set; } = Constants.DEFAULT_BIT_DEPTH;
    public Phase phase { get; set; } = Constants.DEFAULT_PHASE;
    public double f_res { get; set; } = Constants.DEFAULT_F_RES;

    public double bass_boost_gain { get; set; } = Constants.DEFAULT_BASS_BOOST_GAIN;
    public double bass_boost_fc { get; set; } = Constants.DEFAULT_BASS_BOOST_FC;
    public double bass_boost_q { get; set; } = Constants.DEFAULT_BASS_BOOST_GAIN;

    public double treble_boost_gain { get; set; } = Constants.DEFAULT_TREBLE_BOOST_GAIN;
    public double treble_boost_fc { get; set; } = Constants.DEFAULT_TREBLE_BOOST_FC;
    public double treble_boost_q { get; set; } = Constants.DEFAULT_TREBLE_BOOST_Q;

    public dynamic tilt { get; set; }
    public dynamic sound_signature { get; set; }

    public double max_gain { get; set; } = Constants.DEFAULT_MAX_GAIN;
    public double window_size { get; set; } = Constants.DEFAULT_SMOOTHING_WINDOW_SIZE;
    public double treble_window_size { get; set; } = Constants.DEFAULT_TREBLE_SMOOTHING_WINDOW_SIZE;
    public double treble_f_lower { get; set; } = Constants.DEFAULT_TREBLE_F_LOWER;
    public double treble_f_upper { get; set; } = Constants.DEFAULT_TREBLE_F_UPPER;
    public double treble_gain_k { get; set; } = Constants.DEFAULT_TREBLE_GAIN_K;

    public double preamp { get; set; } = Constants.DEFAULT_PREAMP;
    public int thread_count { get; set; } = Environment.ProcessorCount;

    public BatchProcessing()
    {
    }
    public void BatchProcess()
    {
    }

        public static BatchProcessing process_file_wrapper(dynamic @params)
    {
        return process_file(*@params);
    }

    public object process_file(string input_file_path, string output_file_path)
    {
        // Read data from input file
        FrequencyResponse fr = FrequencyResponse.read_from_csv(input_file_path);

        // Copy relative path to output directory
        string output_dir_path = new FileInfo(output_file_path).Directory!.FullName;
        Directory.CreateDirectory(output_dir_path);


        if (standardize_input)
        {
            // Overwrite input data in standard sampling and zero bias
            fr.interpolate();
            fr.center();
            fr.write_to_csv(input_file_path);
        }
        if (ten_band_eq)
            // Ten band eq is a shortcut for setting Fc and Q values to standard 10-band equalizer filters parameters
            fixed_band_eq = true;
        fixed_band_eq_config = PEQ_CONFIGS["10_BAND_GRAPHIC_EQ"];

        if (rockbox && !ten_band_eq)
            throw new ArgumentException("Rockbox configuration requires ten-band eq");
        /*
     // Process and equalize
     fr.process(
         compensation = compensation,
         min_mean_error = true,
         bass_boost_gain = bass_boost_gain,
         bass_boost_fc = bass_boost_fc,
         bass_boost_q = bass_boost_q,
         treble_boost_gain = treble_boost_gain,
         treble_boost_fc = treble_boost_fc,
         treble_boost_q = treble_boost_q,
         tilt = tilt,
         fs = fs[0],
         sound_signature = sound_signature,
         max_gain = max_gain,
         window_size = window_size,
         treble_window_size = treble_window_size,
         treble_f_lower = treble_f_lower,
         treble_f_upper = treble_f_upper,
         treble_gain_k = treble_gain_k);

     fr.write_eqapo_graphic_eq(output_file_path.replace(".csv", " GraphicEQ.txt"), normalize = true, preamp = preamp);

     if(parametric_eq)
         parametric_peqs = fr.optimize_parametric_eq(parametric_eq_config, fs[0], preamp = preamp) if parametric_eq else None
             fr.write_eqapo_parametric_eq(output_file_path.replace(".csv", " ParametricEQ.txt"), parametric_peqs)
     else
             parametric_peqs = None;

     if(fixed_band_eq)
         fixed_band_peq = fr.optimize_fixed_band_eq(
             fixed_band_eq_config, fs[0], preamp = preamp)[0] if fixed_band_eq else None
     fr.write_eqapo_parametric_eq(output_file_path.replace(".csv", " FixedBandEQ.txt"), [fixed_band_peq])
             if(rockbox)
             // Write 10 band eq to Rockbox .cfg file
             fr.write_rockbox_10_band_fixed_eq(output_file_path.replace(".csv", " RockboxEQ.cfg"), fixed_band_peq)
     else
         fixed_band_peq = None


     if (convolution_eq)
     {
         foreach (var _fs in fs)
         {
             if (phase == Phase.minimum || phase == Phase.both)  // Write minimum phase impulse response
             {
                 minimum_phase_fir = fr.minimum_phase_impulse_response(
                     fs = _fs, f_res = f_res, normalize = True, preamp = preamp)
                 minimum_phase_ir = np.tile(minimum_phase_fir, (2, 1)).T
                 sf.write(output_file_path.replace(".csv", f" minimum phase {_fs}Hz.wav"), minimum_phase_ir, _fs, bit_depth);
             }

             if (phase == Phase.linear || phase == Phase.both)  // Write linear phase impulse response
             {
                 linear_phase_fir = fr.linear_phase_impulse_response(
                 fs = _fs, f_res = f_res, normalize = True, preamp = preamp);
                 linear_phase_ir = np.tile(linear_phase_fir, (2, 1)).T;
                 sf.write(
                 output_file_path.replace(".csv", f" linear phase {_fs}Hz.wav"), linear_phase_ir, _fs, bit_depth);
             }
         }
     }

     fr.write_to_csv(output_file_path);


     fr.plot_graph(
         show = False,
         close = True,
         file_path = output_file_path.replace(".csv", ".png")
     );


     fr.write_readme(
         os.path.join(output_dir_path, "README.md"),
         parametric_peqs = parametric_peqs,
         fixed_band_peq = fixed_band_peq);
     */
        return fr;
    }
}
/*
from autoeq.constants import Constants.DEFAULT_MAX_GAIN, Constants.DEFAULT_TREBLE_F_LOWER, Constants.DEFAULT_TREBLE_F_UPPER, \
    Constants.DEFAULT_TREBLE_GAIN_K, Constants.DEFAULT_FS, Constants.DEFAULT_BIT_DEPTH, Constants.DEFAULT_PHASE, Constants.DEFAULT_F_RES, Constants.DEFAULT_BASS_BOOST_GAIN, \
    Constants.DEFAULT_BASS_BOOST_FC, Constants.DEFAULT_BASS_BOOST_Q, Constants.DEFAULT_SMOOTHING_WINDOW_SIZE, \
    Constants.DEFAULT_TREBLE_SMOOTHING_WINDOW_SIZE, PEQ_CONFIGS, Constants.DEFAULT_TREBLE_BOOST_GAIN, Constants.DEFAULT_TREBLE_BOOST_Q, \
    Constants.DEFAULT_TREBLE_BOOST_FC, Constants.DEFAULT_PREAMP
from autoeq.frequency_response import FrequencyResponse


def batch_processing(input_dir=None, output_dir=None, new_only=False, standardize_input=False, compensation=None,
                     parametric_eq=False, fixed_band_eq=False, rockbox=False,
                     ten_band_eq=False, parametric_eq_config=None, fixed_band_eq_config=None, convolution_eq=False,
                     fs=Constants.DEFAULT_FS, bit_depth=Constants.DEFAULT_BIT_DEPTH, phase=Constants.DEFAULT_PHASE, f_res=Constants.DEFAULT_F_RES,
                     bass_boost_gain=Constants.DEFAULT_BASS_BOOST_GAIN, bass_boost_fc=Constants.DEFAULT_BASS_BOOST_FC,
                     bass_boost_q=Constants.DEFAULT_BASS_BOOST_Q, treble_boost_gain=Constants.DEFAULT_TREBLE_BOOST_GAIN,
                     treble_boost_fc=Constants.DEFAULT_TREBLE_BOOST_FC, treble_boost_q=Constants.DEFAULT_TREBLE_BOOST_Q,
                     tilt=None, sound_signature=None, max_gain=Constants.DEFAULT_MAX_GAIN,
                     window_size=Constants.DEFAULT_SMOOTHING_WINDOW_SIZE, treble_window_size=Constants.DEFAULT_TREBLE_SMOOTHING_WINDOW_SIZE,
                     treble_f_lower=Constants.DEFAULT_TREBLE_F_LOWER, treble_f_upper=Constants.DEFAULT_TREBLE_F_UPPER,
                     treble_gain_k=Constants.DEFAULT_TREBLE_GAIN_K, preamp=Constants.DEFAULT_PREAMP, thread_count=0):
    """Parses files in input directory and produces equalization results in output directory."""
    if(not compensation and (parametric_eq or fixed_band_eq or rockbox or ten_band_eq or convolution_eq))
        raise ValueError("Compensation must be specified when equalizing.")

    // Dir paths to absolute
    input_dir = os.path.abspath(input_dir)
    glob_files = glob(os.path.join(input_dir, "**", "*.csv"), recursive=True)
    if(len(glob_files) == 0)
        raise FileNotFoundError(f"No CSV files found in "{input_dir}"")

    if(compensation)
        // Creates FrequencyResponse for compensation data
        compensation_path = os.path.abspath(compensation)
        compensation = FrequencyResponse.read_from_csv(compensation_path)
        compensation.interpolate()
        compensation.center()

    if(bit_depth == 16)
        bit_depth = "PCM_16"
    elif(bit_depth == 24)
        bit_depth = "PCM_24"
    elif(bit_depth == 32)
        bit_depth = "PCM_32"
    else:
        raise ValueError("Invalid bit depth. Accepted values are 16, 24 and 32.")

    if(sound_signature is not None)
        sound_signature = FrequencyResponse.read_from_csv(sound_signature)
        if(len(sound_signature.error) > 0)
            // Error data present, replace raw data with it
            sound_signature.raw = sound_signature.error
        sound_signature.interpolate()
        sound_signature.center()

    if(parametric_eq_config is not None)
        if(type(parametric_eq_config) is str and os.path.isfile(parametric_eq_config))
            // Parametric EQ config is a file path
            with open(parametric_eq_config) as fh:
                parametric_eq_config = yaml.safe_load(fh)
        else:
            if(type(parametric_eq_config) is str)
                parametric_eq_config = [parametric_eq_config]
            parametric_eq_config = [
                PEQ_CONFIGS[config] if type(config) is str else config for config in parametric_eq_config]

    if(fixed_band_eq_config is not None)
        if(os.path.isfile(fixed_band_eq_config))
            // Parametric EQ config is a file path
            with open(fixed_band_eq_config) as fh:
                fixed_band_eq_config = yaml.safe_load(fh)
        else:
            if(fixed_band_eq_config not in PEQ_CONFIGS)
                raise ValueError(
                    f"Unrecognized fixed band eq config "{fixed_band_eq_config}"."
                    f"If this was meant to be a file, the file does not exist.")
            fixed_band_eq_config = PEQ_CONFIGS[fixed_band_eq_config]

    // Prepare list of arguments for all the function calls to generate results.
    n_total = 0
    file_paths = []
    args_list = []
    for input_file_path in glob_files:
        relative_path = os.path.relpath(input_file_path, input_dir)
        output_file_path = os.path.join(output_dir, relative_path) if output_dir else None
        output_file_dir = os.path.split(output_file_path)[0]
        if(not new_only or not os.path.isdir(output_file_dir) or not len(os.listdir(output_file_dir)))
            // Not looking for only new ones or the output directory doesn"t exist or it"s empty
            file_paths.append((input_file_path, output_file_path))
            n_total += 1
            args = (input_file_path, output_file_path, bass_boost_fc, bass_boost_gain, bass_boost_q,
                    treble_boost_fc, treble_boost_gain, treble_boost_q,
                    bit_depth, compensation, convolution_eq, f_res, fixed_band_eq, fs, parametric_eq_config,
                    fixed_band_eq_config, max_gain, window_size, treble_window_size,
                    parametric_eq, phase, rockbox, sound_signature, standardize_input,
                    ten_band_eq, tilt, treble_f_lower, treble_f_upper, treble_gain_k, preamp)
            args_list.append(args)

    if(not thread_count)
        thread_count = multiprocessing.cpu_count()

    with multiprocessing.Pool(thread_count) as pool:
        results = []
        for result in tqdm.tqdm(
                pool.imap_unordered(process_file_wrapper, args_list, chunksize=1), total=len(args_list)):
            results.append(result)
        return results






*/