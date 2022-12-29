using System;
using System.Collections.Generic;

namespace AutoEQ2.Core;

class OptimizationHistory
{
    public DateTime start_time { get; } = DateTime.Now;

    public List<DateTime> time { get; } = new();
    public List<dynamic> loss { get; } = new();
    public List<dynamic> moving_avg_loss { get; } = new();
    public List<dynamic> change_rate { get; } = new();
    public List<dynamic> std { get; } = new();
    public List<dynamic> @params { get; } = new();
}

