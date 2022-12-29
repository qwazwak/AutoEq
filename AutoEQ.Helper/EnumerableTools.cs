using System;
using System.Collections.Generic;
using System.Linq;

namespace AutoEQ2.Helper;

public static class EnumerableTools
{
    //public static IEnumerable<int> GetSortedIndexs<T>(IEnumerable<T> values) => values.Select((v, i) => (v, i)).OrderBy(p => p.v).Select(p => p.i);
    //public static IEnumerable<T> SortByIndexes<T>(IList<T> values, IEnumerable<int> Indexes) => Indexes.Select(i => values[i]);

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

    public static int argmax(this IEnumerable<double> values) => argmax(values, out double _);
    public static int argmax(this IEnumerable<double> values, out double MaxValue)
    {
        using IEnumerator<double> e = values.GetEnumerator();
        if (!e.MoveNext())
            throw new ArgumentException("Enumerable was empty", nameof(values));

        MaxValue = e.Current;
        int iMax = 0;
        int i = 0;

        while (e.MoveNext())
        {
            i++;
            if(e.Current > MaxValue)
            {
                MaxValue = e.Current;
                iMax = i;
            }
        }
        return iMax;
    }
}
