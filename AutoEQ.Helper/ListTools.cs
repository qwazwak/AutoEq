using System.Collections.Generic;

namespace AutoEQ2.Helper;

public static class ListTools
{
    public static IEnumerable<T> EnumerateRange<T>(this IList<T> list, int StartingIndex, int LastIndex, bool ReturnLastToo = true)
    {
        LastIndex = ReturnLastToo ? LastIndex + 1 : LastIndex;
        for (int i = StartingIndex; i < LastIndex; i++)
            yield return list[i];
    }
    public static IEnumerable<T> EnumerateFrom<T>(this IList<T> list, int StartingIndex) => list.EnumerateRange(StartingIndex, list.Count, false);
    public static IEnumerable<int> GetSortingIndexs<T>(this IList<T> list) => list.Select((v, i) => (v, i)).OrderBy(p => p.v).Select(p => p.i);
    public static List<T> SortByIndexes<T>(this IList<T> list, IEnumerable<int> indexs) => indexs.Select(i => list[i]).ToList();
}
