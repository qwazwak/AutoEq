using System;
using System.Collections.Generic;
using System.Linq;

namespace AutoEQ.Helper;

public static class Regressor
{
    /// <summary>
    /// Fits a line to a collection of (x,y) points.
    /// </summary>
    /// <param name="xVals">The x-axis values.</param>
    /// <param name="yVals">The y-axis values.</param>
    /// <param name="rSquared">The r^2 value of the line.</param>
    /// <param name="yIntercept">The y-intercept value of the line (i.e. y = ax + b, yIntercept is b).</param>
    /// <param name="slope">The slop of the line (i.e. y = ax + b, slope is a).</param>
    public static void LinearRegression(IEnumerable<double> xVals, IEnumerable<double> yVals, out double rSquared, out double yIntercept, out double slope) => LinearRegression(xVals.Zip(yVals), out rSquared, out yIntercept, out slope);
    /// <summary>
    /// Fits a line to a collection of (x,y) points.
    /// </summary>
    /// <param name="xVals">The x-axis values.</param>
    /// <param name="yVals">The y-axis values.</param>
    /// <param name="rSquared">The r^2 value of the line.</param>
    /// <param name="yIntercept">The y-intercept value of the line (i.e. y = ax + b, yIntercept is b).</param>
    /// <param name="slope">The slop of the line (i.e. y = ax + b, slope is a).</param>
    public static void LinearRegression(IEnumerable<(double x, double y)> xyVals, out double rSquared, out double yIntercept, out double slope)
    {
        double sumOfX = 0;
        double sumOfY = 0;
        double sumOfXSq = 0;
        double sumOfYSq = 0;
        double sumCodeviates = 0;
        int Count = 0;
        foreach ((double x, double y) in xyVals)
        {
            Count++;
            sumCodeviates += x * y;
            sumOfX += x;
            sumOfY += y;
            sumOfXSq += x * x;
            sumOfYSq += y * y;
        }

        double meanX = sumOfX / Count;
        double meanY = sumOfY / Count;
        double ssX = sumOfXSq - (sumOfX * sumOfX / Count);
        //double ssY = sumOfYSq - (sumOfY * sumOfY / Count);

        double rNumerator = (Count * sumCodeviates) - (sumOfX * sumOfY);
        double rDenom = ((Count * sumOfXSq) - (sumOfX * sumOfX)) * ((Count * sumOfYSq) - (sumOfY * sumOfY));
        double sCo = sumCodeviates - (sumOfX * sumOfY / Count);

        double dblR = rNumerator / Math.Sqrt(rDenom);

        rSquared = dblR * dblR;
        yIntercept = meanY - (sCo / ssX * meanX);
        slope = sCo / ssX;
    }
}

public static class SumAndCountExtensions
{
    public static void SumAndCount(this IEnumerable<double> values, out double Sum, out int Count) => (Sum, Count) = SumAndCount(values);
    public static (double Sum, int Count) SumAndCount(this IEnumerable<double> values) => ((double Sum, int Count))SumAndCountCore((IEnumerable<dynamic>)values);

    internal static void SumAndCountCore<T>(this IEnumerable<T> values, out T Sum, out int Count) => (Sum, Count) = values.SumAndCountCore();
    internal static void SumAndCountCore(this IEnumerable<dynamic> values, out dynamic Sum, out int Count) => (Sum, Count) = values.SumAndCountCore();
    internal static (T Sum, int Count) SumAndCountCore<T>(this IEnumerable<T> values) => ((T Sum, int Count))((IEnumerable<dynamic>)values).SumAndCountCore();
    internal static (dynamic Sum, int Count) SumAndCountCore(this IEnumerable<dynamic> values)
    {
        if (!values.TryGetNonEnumeratedCount(out int Count))
        {
            if (values is IEnumerable<double> dubs) 
                return (dubs.Sum(), Count);
            if (values is IEnumerable<float> floats) 
                return (floats.Sum(), Count);
        }
        dynamic Sum = default;
        Count = 0;
        foreach (dynamic i in values)
        {
            Sum += i;
            Count++;
        }
        return (Sum, Count);
    }
}
