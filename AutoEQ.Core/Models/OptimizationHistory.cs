using System.Collections.Generic;
using System;
using System.Diagnostics;
using System.Linq;
using AutoEQ.Helper;

namespace AutoEQ.Core.Models;

public class OptimizationHistory
{
    public const int n = 8;
    public Stopwatch SW { get; } = new();
    public DateTime start_time { get; } = DateTime.Now;
    public List<OptmizationPoint> Points { get; } = new();
    /*
    public List<object> time { get; } = new();
    public List<object> loss { get; } = new();
    public List<object> moving_avg_loss { get; } = new();
    public List<object> change_rate { get; } = new();
    public List<object> std { get; } = new();
    public List<object> @params { get; } = new();*/

    //self.history.time.append(t)
    public IEnumerable<OptmizationPoint> PointsReverse => Points.Reverse<OptmizationPoint>();
    public IEnumerable<OptmizationPoint> LastN(int Count) => PointsReverse.Take(Count);
    public IEnumerable<OptmizationPoint> LastN() => LastN(n);
    public IEnumerable<OptmizationPoint> LastFrac(double fraction) => LastN((int)Math.Ceiling(n * fraction));

    public double LastStd(double fraction) => MathEx.StdDiv(LastFrac(fraction).Select(p => p.loss));

    public class OptmizationPoint
    {
        public TimeSpan TimeTaken { get; init; }
        public TimeSpan time => TimeTaken;
        public double loss { get; init; }
        public double moving_avg_loss { get; init; }
        public double change_rate { get; init; }
        public double std { get; init; }
        public dynamic @params { get; init; }
    }
}
