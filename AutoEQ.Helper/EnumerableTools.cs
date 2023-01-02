using System;
using System.Collections.Generic;
using System.Linq;

namespace AutoEQ.Helper;

public static class EnumerableTools
{
    //public static IEnumerable<int> GetSortedIndexs<T>(IEnumerable<T> values) => values.Select((v, i) => (v, i)).OrderBy(p => p.v).Select(p => p.i);
    //public static IEnumerable<T> SortByIndexes<T>(IList<T> values, IEnumerable<int> Indexes) => Indexes.Select(i => values[i]);
    //.Select(i => Math.Abs(i - Freq))
    public static IEnumerable<T> SelectWhere<T>(this IEnumerable<T> values, IEnumerable<bool> Selectors)
    {
        foreach ((T value, bool selection) in values.Zip(Selectors))
            if(selection) yield return value;
    }
    public static IEnumerable<int> IndexWhere<T>(this IEnumerable<T> values, Func<T, bool> Predicte) => values.Select((v, index) => (v, index)).Where(p => Predicte.Invoke(p.v)).Select(p => p.index);

    public static int IndexClosestTo(this IEnumerable<double> values, double Target) => values.IndexClosestTo(Target, out double _);
    public static int IndexClosestTo(this IEnumerable<double> values, double Target, out double ClosestValue) => values.Select(i => Math.Abs(i - Target)).argMin(out ClosestValue);

    public static double Average<T>(IEnumerable<double> values)
    {
        using IEnumerator<double> e = values.GetEnumerator();
        double Sum = 0;
        int i = 0;
        while (e.MoveNext())
        {
            i++;
            Sum += e.Current;
        }
        return Sum / i;
    }

    public static int argmax<T>(this IEnumerable<T> values) where T : IComparable<T> => argmax(values, out T _);
    public static int argmax<T>(this IEnumerable<T> values, out T MaxValue) where T : IComparable<T>
    {
        using IEnumerator<T> e = values.GetEnumerator();
        if (!e.MoveNext())
            throw new ArgumentException("Enumerable was empty", nameof(values));

        MaxValue = e.Current;
        int iMax = 0;
        int i = 0;

        while (e.MoveNext())
        {
            i++;
            if(e.Current.CompareTo(MaxValue) < 0)
            {
                MaxValue = e.Current;
                iMax = i;
            }
        }
        return iMax;
    }
    public static int argMin<T>(this IEnumerable<T> values) where T : IComparable<T> => argMin(values, out T _);
    public static int argMin<T>(this IEnumerable<T> values, out T MaxValue) where T : IComparable<T>
    {
        using IEnumerator<T> e = values.GetEnumerator();
        if (!e.MoveNext())
            throw new ArgumentException("Enumerable was empty", nameof(values));

        MaxValue = e.Current;
        int iMax = 0;
        int i = 0;

        while (e.MoveNext())
        {
            i++;
            if(e.Current.CompareTo(MaxValue) > 0)
            {
                MaxValue = e.Current;
                iMax = i;
            }
        }
        return iMax;
    }
}
