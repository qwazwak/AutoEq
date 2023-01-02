using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

namespace AutoEQ.Helper;

public static class MathEx
{
    private static readonly ImmutableArray<Type> NumberTypes = FloatTypes.AddRange(IntegerTypes);
    private static readonly ImmutableArray<Type> FloatTypes = ImmutableArray.Create(
        typeof(float),
        typeof(double),
        typeof(decimal));
    private static readonly ImmutableArray<Type> IntegerTypes = ImmutableArray.Create(
        typeof(sbyte),
        typeof(byte),
        typeof(short),
        typeof(ushort),
        typeof(int),
        typeof(uint),
        typeof(long),
        typeof(ulong),
        typeof(nint),
        typeof(nuint));
    public static double DotProduct(this IEnumerable<double> A, IEnumerable<double> B) => A.MultiplyPairs(B).Sum();
    public static double DotProductSingle(this IEnumerable<double> A) => A.Sum(a => a * a);
    public static IEnumerable<double> MultiplyPairs(this IEnumerable<double> A, IEnumerable<double> B) => A.Zip(B, (a, b) => a * b);
    
    public static dynamic Mean(IEnumerable<double> arr) => MeanCore((IEnumerable<dynamic>)arr);
    public static dynamic MeanCore(IEnumerable<dynamic> arr)
    {
        arr.SumAndCountCore(out dynamic Sum, out int Count);
        return Sum / Count;
    }

    private static double Variance(IEnumerable<double> arr, out double Result, int ddof = 0) => Result = Variance(arr, ddof);
    private static double Variance(IEnumerable<double> arr, int ddof = 0)
    {
        // Make this warning show up on top.
        if (ddof >= arr.Count())
            throw new DivideByZeroException("Degrees of freedom <= 0 for slice");

        // Compute the mean.
        double arrmean = arr.Average();

        // Compute degrees of freedom and make sure it is not negative.
        int count = Math.Max(arr.Count() - ddof, 0);

        // Compute sum of squared deviations from mean
        // Note that x may not be inexact and that we need it to be an array,
        // not a scalar.
        double SumOfSquares = arr.Sum(i => Math.Pow(i - arrmean, 2));

        // divide by degrees of freedom
        return SumOfSquares / count;
    }
    public static double StdDiv(IEnumerable<double> a, out double Result, int ddof = 0) => Result = StdDiv(a, ddof: ddof);
    public static double StdDiv(IEnumerable<double> a, int ddof = 0) => Math.Sqrt(Variance(a, ddof: ddof));
}
