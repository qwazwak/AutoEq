using AutoEQ.Core;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Immutable;
using System.IO;
using System.Text;

namespace UnitTests;
[TestClass]
public class UnitTest1
{
    private static readonly DirectoryInfo TempFilesRootDir = new(Path.GetTempPath());
    private DirectoryInfo root = null!;
    private DirectoryInfo input = null!;
    private DirectoryInfo output = null!;
    private FileInfo compensation = null!;
    private FileInfo sound_signature = null!;
    private string _root => root.FullName;
    private string _input => input.FullName;
    private string _output => input.FullName;
    private string _compensation => compensation.FullName;
    private string _sound_signature => sound_signature.FullName;

    private static readonly ImmutableArray<string> InputFileLines = ImmutableArray.Create(
        "frequency,raw\n20,2\n50,2\n200,0\n1000,1\n3000,10\n10000,0\n20000,-15");

    [ClassInitialize]
    public void Setup()
    {
        root = TempFilesRootDir.CreateSubdirectory(Guid.NewGuid().ToString());
        root.Create();
        input = root.CreateSubdirectory("input");
        output = root.CreateSubdirectory("output");
        for (int i = 0; i < 3; i++)
        {
            DirectoryInfo path = input.CreateSubdirectory($"Headphone {i}");
            path.Create();
            FileInfo file = new(Path.Join(path.FullName, $"Headphone {i}.csv"));
            File.WriteAllLines(file.FullName, InputFileLines, Encoding.UTF8);
        }
        compensation = new(Path.Join(root.FullName, "compensation.csv"));
        using(FileStream fh = compensation.OpenWrite())
        {
            FrequencyResponse fr = new(
                "compensation",
                frequency: new double[] { 20, 50, 200, 1000, 3000, 10000, 20000 },
                raw: new double[] { 6, 6, -1, 0, 8, 1, -10 }
                );
            fr.interpolate(pol_order = 2);
            fr.smoothen_fractional_octave(window_size = 2, treble_window_size = 2);
            fr.center();
            fr.write_to_csv(_compensation);
        }
        sound_signature = new(root.FullName + Path.DirectorySeparatorChar + "sound_signature.csv");
        using(FileStream fh = sound_signature.OpenWrite())
        {
            using StreamWriter sw = new(fh);
            sw.WriteLine("frequency,raw\n20.0,0\n10000,0.0\n20000,3");
        }
    }
    [ClassCleanup]
    public void Teardown()
    {
        //shutil.rmtree(self._root)
    }
    [TestMethod]
    public void test_batch_processing()
    {
        Assert.IsTrue(Directory.Exists(Path.Join(_input, "Headphone 1", "Headphone 1.csv")));
        Assert.IsTrue(Directory.Exists(Path.Join(_input, "Headphone 2", "Headphone 2.csv")));
        /*
            frs = batch_processing(
                input_dir = self._input, output_dir = self._output, standardize_input = True, compensation = self._compensation,
                parametric_eq = True, fixed_band_eq = True, rockbox = True,
                ten_band_eq = True,
                parametric_eq_config =["4_PEAKING_WITH_LOW_SHEL$", PEQ_CONFIGS["4_PEAKING_WITH_HIGH_SHELF"]],
                fixed_band_eq_config = None, convolution_eq = True,
                fs =[44100, 48000], bit_depth = DEFAULT_BIT_DEPTH, phase = "both", f_res = DEFAULT_F_RES,
                bass_boost_gain = DEFAULT_BASS_BOOST_GAIN, bass_boost_fc = DEFAULT_BASS_BOOST_FC,
                bass_boost_q = DEFAULT_BASS_BOOST_Q, treble_boost_gain = DEFAULT_TREBLE_BOOST_GAIN,
                treble_boost_fc = DEFAULT_TREBLE_BOOST_FC, treble_boost_q = DEFAULT_TREBLE_BOOST_Q,
                tilt = -0.2, sound_signature = self._sound_signature,
                max_gain = DEFAULT_MAX_GAIN,
                window_size = DEFAULT_SMOOTHING_WINDOW_SIZE, treble_window_size = DEFAULT_TREBLE_SMOOTHING_WINDOW_SIZE,
                treble_f_lower = DEFAULT_TREBLE_F_LOWER, treble_f_upper = DEFAULT_TREBLE_F_UPPER,
                treble_gain_k = DEFAULT_TREBLE_GAIN_K, preamp = -1.0, thread_count = 1
            )
            self.assertEqual(len(frs), 2)

            self.assertTrue(self._output.joinpath("Headphone 1", "Headphone 1.png").exists())

            # CSV file
        self.assertTrue(self._output.joinpath("Headphone 1", "Headphone 1.csv").exists())
            df = pd.read_csv(self._output.joinpath("Headphone 1", "Headphone 1.csv"))
            columns = "frequency,raw,error,smoothed,error_smoothed,equalization,parametric_eq,fixed_band_eq," \
                      "equalized_raw,equalized_smoothed,target".split(",")
            self.assertEqual(list(df.columns), columns)
            self.assertEqual(df.size, 695 * len(columns))

            # Graphic equalizer
        self.assertTrue(self._output.joinpath("Headphone 1", "Headphone 1 GraphicEQ.txt").exists())
            with open(self._output.joinpath("Headphone 1", "Headphone 1 GraphicEQ.txt")) as fh:
                self.assertRegexpMatches(fh.read().strip() + "; ", r"GraphicEQ: \d{2,5} (-?\d(\.\d+)?; )+")

            # Fixed band equalizer
        self.assertTrue(self._output.joinpath("Headphone 1", "Headphone 1 FixedBandEq.txt").exists())
            with open(self._output.joinpath("Headphone 1", "Headphone 1 FixedBandEq.txt")) as fh:
                lines = fh.read().strip().split("\n")
            self.assertTrue(re.match(r"Preamp: -?\d+(\.\d+)? dB", lines[0]))
            for line in lines[1:]:
                self.assertRegexpMatches(line, r"Filter \d{1,2}: ON PK Fc \d{2,5} Hz Gain -?\d(\.\d+)? dB Q 1.41")

            # Parametric equalizer
            self.assertTrue(self._output.joinpath("Headphone 1", "Headphone 1 ParametricEq.txt").exists())
            with open(self._output.joinpath("Headphone 1", "Headphone 1 ParametricEq.txt")) as fh:
                lines = fh.read().strip().split("\n")
            self.assertTrue(re.match(r"Preamp: -?\d+(\.\d+)? dB", lines[0]))
            for line in lines[1:]:
                self.assertRegexpMatches(
                    line, r"Filter \d{1,2}: ON (PK|LS|HS) Fc \d{2,5} Hz Gain -?\d(\.\d+)? dB Q \d(\.\d+)?")

            # Convolution (FIR) filters
                for phase in ["minimum", "linear"]:
                for fs in [44100, 48000]:
                    fp = self._output.joinpath("Headphone 1", $"Headphone 1 {phase} phase {fs}Hz.wav")
                    self.assertTrue(fp.exists())
                    # Frequency resolution is 10, 2 channels, 16 bits per sample, 8 bits per byte
                    # Real file size has headers
                    min_size = fs / 10 * 2 * 16 / 8
                    self.assertGreater(os.stat(fp).st_size, min_size)

            # README
            self.assertTrue(self._output.joinpath("Headphone 1", "README.md").exists())
            with open(self._output.joinpath("Headphone 1", "README.md")) as fh:
                s = fh.read().strip()
            self.assertTrue("# Headphone 1" in s)
            self.assertTrue("### Parametric EQs" in s)
            self.assertTrue("### Fixed Band EQs" in s)
            self.assertTrue("### Graphs" in s)*/
    }
}
