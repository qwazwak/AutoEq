﻿using AutoEQ2.Core.Models.Eums;
using System.Collections.Immutable;
using System.Linq;
using System.Text.RegularExpressions;

namespace AutoEQ2.Core;

public class Constants
{

    public const double DEFAULT_F_MIN = 20.0;
    public const double DEFAULT_F_MAX = 20000.0;
    public const double DEFAULT_STEP = 1.01;

    public const double DEFAULT_MAX_GAIN = 6.0;
    public const double DEFAULT_TREBLE_F_LOWER = 6000.0;
    public const double DEFAULT_TREBLE_F_UPPER = 8000.0;
    public const double DEFAULT_TREBLE_MAX_GAIN = 6.0;
    public const double DEFAULT_TREBLE_GAIN_K = 1.0;

    public const double DEFAULT_SMOOTHING_WINDOW_SIZE = 1 / 12;
    public const double DEFAULT_SMOOTHING_ITERATIONS = 1;
    public const double DEFAULT_TREBLE_SMOOTHING_F_LOWER = 100.0;
    public const double DEFAULT_TREBLE_SMOOTHING_F_UPPER = 10000.0;
    public const double DEFAULT_TREBLE_SMOOTHING_WINDOW_SIZE = 2.0;
    public const double DEFAULT_TREBLE_SMOOTHING_ITERATIONS = 1;

    public const double DEFAULT_FS = 44100;
    public const int DEFAULT_BIT_DEPTH = (int)DEFAULT_BIT_DEPTH_Enum;
    public const BitDepth DEFAULT_BIT_DEPTH_Enum = BitDepth.PCM_16;
    public const Phase DEFAULT_PHASE = Phase.Minimum;
    public const double DEFAULT_F_RES = 10.0;

    public const double DEFAULT_TILT = 0.0;
    public const double DEFAULT_BASS_BOOST_GAIN = 0.0;
    public const double DEFAULT_BASS_BOOST_FC = 105.0;
    public const double DEFAULT_BASS_BOOST_Q = 0.7;
    public const double DEFAULT_TREBLE_BOOST_GAIN = 0.0;
    public const double DEFAULT_TREBLE_BOOST_FC = 10000.0;
    public const double DEFAULT_TREBLE_BOOST_Q = 0.7;

    public const double DEFAULT_PEQ_OPTIMIZER_MIN_F = 20.0;
    public const double DEFAULT_PEQ_OPTIMIZER_MAX_F = 20000.0;
    public const dynamic? DEFAULT_PEQ_OPTIMIZER_MAX_TIME = None;
    public const dynamic? DEFAULT_PEQ_OPTIMIZER_TARGET_LOSS = None;
    public const dynamic? DEFAULT_PEQ_OPTIMIZER_MIN_CHANGE_RATE = None;
    public const double DEFAULT_PEQ_OPTIMIZER_MIN_STD = 0.002;

    public const double DEFAULT_FIXED_BAND_FILTER_MIN_GAIN = -12.0;
    public const double DEFAULT_FIXED_BAND_FILTER_MAX_GAIN = 12.0;

    public const double DEFAULT_PEAKING_FILTER_MIN_FC = 20.0;
    public const double DEFAULT_PEAKING_FILTER_MAX_FC = 10000.0;
    /// <summary>
    /// AUNBandEq has maximum bandwidth of 5 octaves which is Q of 0.182479
    /// </summary>
    public const double DEFAULT_PEAKING_FILTER_MIN_Q = 0.18248;
    public const double DEFAULT_PEAKING_FILTER_MAX_Q = 6.0;
    public const double DEFAULT_PEAKING_FILTER_MIN_GAIN = -20.0;
    public const double DEFAULT_PEAKING_FILTER_MAX_GAIN = 20.0;

    public const double DEFAULT_SHELF_FILTER_MIN_FC = 20.0;
    public const double DEFAULT_SHELF_FILTER_MAX_FC = 10000.0;
    /// <summary>
    /// Shelf filters start to overshoot below 0.4
    /// </summary>
    public const double DEFAULT_SHELF_FILTER_MIN_Q = 0.4;
    /// <summary>
    ///  Shelf filters start to overshoot above 0.7
    /// </summary>
    public const double DEFAULT_SHELF_FILTER_MAX_Q = 0.7;
    public const double DEFAULT_SHELF_FILTER_MIN_GAIN = -20.0;
    public const double DEFAULT_SHELF_FILTER_MAX_GAIN = 20.0;

    public const double DEFAULT_BIQUAD_OPTIMIZATION_F_STEP = 1.02;

    public const double DEFAULT_MAX_SLOPE = 18.0;
    public const double DEFAULT_PREAMP = 0.0;
    /// <summary>
    /// Produces 127 samples with greatest frequency of 19871
    /// </summary>
    public const double DEFAULT_GRAPHIC_EQ_STEP = 1.0563;

    public static class SemiConstants
    {

    }

   // public static readonly string ROOT_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)));
//    public const string MOD_REGEX_STR = " \((sample|serial number) [a-zA-Z0-9\-]+\)$";
    public static readonly Regex MOD_REGEX = new(@" \((sample|serial number) [a-zA-Z0-9\-]+\)$");
    public readonly ImmutableArray<string> DBS = ImmutableArray.Create("crinacle", "headphonecom", "innerfidelity", "oratory1990", "referenceaudioanalyzer", "rtings");


    public static readonly ImmutableArray<double> HARMAN_ONEAR_PREFERENCE_FREQUENCIES = ImmutableArray.Create(20.0, 21.0, 22.0, 24.0, 25.0, 27.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0, 43.0, 45.0, 48.0, 50.0, 53.0, 56.0, 60.0, 63.0, 67.0, 71.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 106.0, 112.0, 118.0, 125.0, 132.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 212.0, 224.0, 236.0, 250.0, 265.0, 280.0, 300.0, 315.0, 335.0, 355.0, 375.0, 400.0, 425.0, 450.0, 475.0, 500.0, 530.0, 560.0, 600.0, 630.0, 670.0, 710.0, 750.0, 800.0, 850.0, 900.0, 950.0, 1000.0, 1060.0, 1120.0, 1180.0, 1250.0, 1320.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0, 2120.0, 2240.0, 2360.0, 2500.0, 2650.0, 2800.0, 3000.0, 3150.0, 3350.0, 3550.0, 3750.0, 4000.0, 4250.0, 4500.0, 4750.0, 5000.0, 5300.0, 5600.0, 6000.0, 6300.0, 6700.0, 7100.0, 7500.0, 8000.0, 8500.0, 9000.0, 9500.0, 10000.0, 10600.0, 11200.0, 11800.0, 12500.0, 13200.0, 14000.0, 15000.0, 16000.0, 17000.0, 18000.0, 19000.0, 20000.0);
public static readonly ImmutableArray<double> HARMAN_INEAR_PREFENCE_FREQUENCIES = ImmutableArray.Create(20.0, 21.2, 22.4, 23.6, 25.0, 26.5, 28.0, 30.0, 31.5, 33.5, 35.5, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0, 53.0, 56.0, 60.0, 63.0, 67.0, 71.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 106.0, 112.0, 118.0, 125.0, 132.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 212.0, 224.0, 236.0, 250.0, 265.0, 280.0, 300.0, 315.0, 335.0, 355.0, 375.0, 400.0, 425.0, 450.0, 475.0, 500.0, 530.0, 560.0, 600.0, 630.0, 670.0, 710.0, 750.0, 800.0, 850.0, 900.0, 950.0, 1000.0, 1060.0, 1120.0, 1180.0, 1250.0, 1320.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0, 2120.0, 2240.0, 2360.0, 2500.0, 2650.0, 2800.0, 3000.0, 3150.0, 3350.0, 3550.0, 3750.0, 4000.0, 4250.0, 4500.0, 4750.0, 5000.0, 5300.0, 5600.0, 6000.0, 6300.0, 6700.0, 7100.0, 7500.0, 8000.0, 8500.0, 9000.0, 9500.0, 10000.0, 10600.0, 11200.0, 11800.0, 12500.0, 13200.0, 14000.0, 15000.0, 16000.0, 17000.0, 18000.0, 19000.0, 20000.0);

    public const double PREAMP_HEADROOM = 0.2;
    public record PEQ_CONFIG(PEQ PEQ,  (Optimizer, double)? Optimizer, ImmutableArray<PEQ_CONFIG.Filter> Filters)
    {
        public record Filter(FilterType Type, double? FC, double? Q);
    }
    
    public static readonly ImmutableArray<PEQ_CONFIG> PEQ_CONFIGS = ImmutableArray.Create(
        new PEQ_CONFIG(PEQ._10_BAND_GRAPHIC_EQ,(Optimizer.MinStd, 0.01), Enumerable.Range(0, 10).Select(i => new PEQ_CONFIG.Filter(FilterType.PEAKING, 31.25 * Math.Pow(2, i), Math.Sqrt(2))).ToImmutableArray())
        );
   /* "10_BAND_GRAPHIC_EQ": {
        "optimizer": {"min_std": 0.01},
        "filters": [{"fc": 31.25 * 2 ** i, "q": math.sqrt(2), "type": "PEAKING"} for i in range(10)]
    },*/
   /*
    "10_PEAKING": {
        "filters": [{"type": "PEAKING"}] * 10
    },
    "8_PEAKING_WITH_SHELVES": {
        "optimizer": {
            "min_std": 0.008
        },
        "filters": [{
            "type": "LOW_SHEL$",
            "fc": 105,
            "q": 0.7
        }, {
            "type": "HIGH_SHEL$",
            "fc": 10e3,
            "q": 0.7
        }] + [{"type": "PEAKING"}] * 8
    },
    "4_PEAKING_WITH_LOW_SHEL$": {
        "optimizer": {
            "max_$": 10000,
        },
        "filters": [{
            "type": "LOW_SHEL$",
            "fc": 105,
            "q": 0.7
        }] + [{"type": "PEAKING"}] * 4
    },
    "4_PEAKING_WITH_HIGH_SHEL$": {
        "filters": [{
            "type": "HIGH_SHEL$",
            "fc": 10000,
            "q": 0.7
        }] + [{"type": "PEAKING"}] * 4
    },
}*/
}
