using System;
using CsvHelper.Configuration.Attributes;

namespace AutoEQ.Core.Models;

public abstract class ResponsePoint : IComparable<ResponsePoint>
{
    public virtual double Frequency { get; set; }

    public virtual double? Raw { get; set; }

    public virtual double? Smoothed { get; set; }

    public virtual double? Error { get; set; }

    public virtual double? Error_smoothed { get; set; }

    public virtual double? Equalization { get; set; }

    public virtual double? Parametric_eq { get; set; }

    public virtual double? Fixed_band_eq { get; set; }

    public virtual double? Equalized_raw { get; set; }

    public virtual double? Equalized_smoothed { get; set; }

    public virtual double? Target { get; set; }

    int IComparable<ResponsePoint>.CompareTo(ResponsePoint? other) => other == null ? throw new NullReferenceException("Cannot compare to null object") : CompareTo(other);
    public virtual int CompareTo(ResponsePoint other) => Frequency.CompareTo(other.Frequency);
}

public class CSVResponsePoint : ResponsePoint
{
    [Name("frequency")]
    public override double Frequency { get; set; }

    [Name("raw")]
    public override double? Raw { get; set; }

    [Name("smoothed")]
    public override double? Smoothed { get; set; }

    [Name("error")]
    public override double? Error { get; set; }

    [Name("error_smoothed")]
    public override double? Error_smoothed { get; set; }

    [Name("equalization")]
    public override double? Equalization { get; set; }

    [Name("parametric_eq")]
    public override double? Parametric_eq { get; set; }

    [Name("fixed_band_eq")]
    public override double? Fixed_band_eq { get; set; }

    [Name("equalized_raw")]
    public override double? Equalized_raw { get; set; }

    [Name("equalized_smoothed")]
    public override double? Equalized_smoothed { get; set; }

    [Name("target")]
    public override double? Target { get; set; }
}
