using System;

namespace AutoEQ.Core;

public class OptimizationFinished : Exception
{
    public OptimizationFinished() : base()
    {
    }

    public OptimizationFinished(string? message) : base(message)
    {
    }

    public OptimizationFinished(string? message, Exception? innerException) : base(message, innerException)
    {
    }
}
