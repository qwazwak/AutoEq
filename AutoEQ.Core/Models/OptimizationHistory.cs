using System.Collections.Generic;
using System;

namespace AutoEQ.Core.Models;

public record OptimizationHistory(List<object> time, List<object> loss, List<object> moving_avg_loss, List<object> change_rate, List<object> std, List<object> @params)
{
    public DateTime start_time { get; } = DateTime.Now;
}
