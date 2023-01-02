using System;
using System.Collections.Generic;
using System.Linq;

namespace AutoEQ.Helper;
public static class ListTools
{
    public static void SubZip(this IList<double> list, IList<double> vals)
    {
        for (int i = 0; i < list.Count; i++)
            list[i] -= vals[i];
    }
    public static void SubEach(this IList<double> list, double val)
    {
        for (int i = 0; i < list.Count; i++)
            list[i] -= val;
    }
    public static void AddEach(this IList<double> list, double val)
    {
        for (int i = 0; i < list.Count; i++)
            list[i] += val;
    }
    public static void MultEach(this IList<double> list, double val)
    {
        for (int i = 0; i < list.Count; i++)
            list[i] *= val;
    }
    public static void MultEach(this double[,] list, double val)
    {
        int L0 = list.GetLength(0);
        int L1 = list.GetLength(1);
        for (int i = 0; i < L0; i++)
            for (int j = 0; j < L1; j++)
                list[i, j] *= val;
    }

    public static void ApplyEach<T>(this IList<T> list, Func<T, T> func)
    {
        for (int i = 0; i < list.Count; i++)
            list[i] = func.Invoke(list[i]);
    }
    public static void SetRange<T>(this IList<T> list, Range range, T Value)
    {
        int i = range.Start.Value;
        Action IncI = range.Start.Value <= range.End.Value ? () => i++ : () => i--;
        for (; i < range.End.Value; IncI())
            list[i] = Value;
    }
    public static IEnumerable<T> EnumerateRange<T>(this IList<T> list, int StartingIndex, int LastIndex, bool ReturnLastToo = true)
    {
        LastIndex = ReturnLastToo ? LastIndex + 1 : LastIndex;
        for (int i = StartingIndex; i < LastIndex; i++)
            yield return list[i];
    }
    public static IEnumerable<T> EnumerateRangeReverse<T>(this IList<T> list, int LastIndex, int FirstIndex, bool ReturnFirstToo = true)
    {
        FirstIndex = ReturnFirstToo ? FirstIndex - 1 : FirstIndex;
        for (int i = LastIndex; i > FirstIndex; i--)
            yield return list[i];
    }
    public static IEnumerable<T> EnumerateFrom<T>(this IList<T> list, int StartingIndex) => list.EnumerateRange(StartingIndex, list.Count, false);
    public static IEnumerable<int> GetSortingIndexs<T>(this IList<T> list) => list.Select((v, i) => (v, i)).OrderBy(p => p.v).Select(p => p.i);
    public static List<T> SortByIndexes<T>(this IList<T> list, IEnumerable<int> indexs) => list.SelectIndexes(indexs).ToList();
    public static IEnumerable<T> SelectIndexes<T>(this IList<T> list, IEnumerable<int> indexs) => indexs.Select(i => list[i]);
}
