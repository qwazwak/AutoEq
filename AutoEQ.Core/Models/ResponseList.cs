using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AutoEQ.Helper;

namespace AutoEQ.Core.Models;

public class ResponseList : ICollection<ResponsePoint>
{
    private readonly List<ResponsePoint> Core = new();
    public IList<double> Frequency { get; }
    public IList<double> Raw { get; }
    public IList<double> Smoothed { get; }
    public IList<double> Error { get; }
    public IList<double> Error_smoothed { get; }
    public IList<double> Equalization { get; }
    public IList<double> Parametric_eq { get; }
    public IList<double> Fixed_band_eq { get; }
    public IList<double> Equalized_raw { get; }
    public IList<double> Equalized_smoothed { get; }
    public IList<double> Target { get; }


    public ResponseList(IEnumerable<ResponsePoint> values) : this()
    {
        AddRange(values);
    }

    public ResponseList()
    {
        Frequency = new IListWrapper<ResponsePoint, double>(Core, rp => rp.Frequency, (f, rp) => rp.Frequency = f);
        Raw = new IListWrapper<ResponsePoint, double>(Core, rp => rp.Raw, (f, rp) => rp.Raw = f);
        Smoothed = new IListWrapper<ResponsePoint, double>(Core, rp => rp.Smoothed, (f, rp) => rp.Smoothed = f);
        Error = new IListWrapper<ResponsePoint, double>(Core, rp => rp.Error, (f, rp) => rp.Error = f);
        Error_smoothed = new IListWrapper<ResponsePoint, double>(Core, rp => rp.Error_smoothed, (f, rp) => rp.Error_smoothed = f);
        Equalization = new IListWrapper<ResponsePoint, double>(Core, rp => rp.Equalization, (f, rp) => rp.Equalization = f);
        Parametric_eq = new IListWrapper<ResponsePoint, double>(Core, rp => rp.Parametric_eq, (f, rp) => rp.Parametric_eq = f);
        Fixed_band_eq = new IListWrapper<ResponsePoint, double>(Core, rp => rp.Fixed_band_eq, (f, rp) => rp.Fixed_band_eq = f);
        Equalized_raw = new IListWrapper<ResponsePoint, double>(Core, rp => rp.Equalized_raw, (f, rp) => rp.Equalized_raw = f);
        Equalized_smoothed = new IListWrapper<ResponsePoint, double>(Core, rp => rp.Equalized_smoothed, (f, rp) => rp.Equalized_smoothed = f);
        Target = new IListWrapper<ResponsePoint, double>(Core, rp => rp.Target, (f, rp) => rp.Target = f);
    }
    public static async Task<ResponseList> InitAsync(IAsyncEnumerable<ResponsePoint> values)
    {
        ResponseList result = new();
        await result.AddRangeAsync(values);
        return result;
    }

    public bool IsSorted { get; private set; }

    private void Sort()
    {
        Core.Sort();
        IsSorted = true;
    }
    private void SortIfNeeded()
    {
        if (!IsSorted)
            Sort();
    }
    private ResponsePoint? FindByFrequency(double Frequency, bool ShouldSort = true)
    {
        if (IsSorted)
            return Search(Frequency);
        else if (ShouldSort)
        {
            Sort();
            return Search(Frequency);
        }
        else
            return Core.FirstOrDefault(f => f.Frequency == Frequency, null!);

        ResponsePoint? Search(double Frequency)
        {
            for (int i = 0; i < Core.Count; i++)
            {
                ResponsePoint r = Core[i];
                if (r.Frequency == Frequency)
                    return r;
                else if (r.Frequency > Frequency)
                    break;
            }
            return null;
        }
    }

    public ResponsePoint this[int index] { get => ((IList<ResponsePoint>)Core)[index]; set => ((IList<ResponsePoint>)Core)[index] = value; }
    //public ResponsePoint this[double frequency] { get => FindByFrequency(frequency); set => ((IList<ResponsePoint>)Core)[index] = value; }

    public int Count => Core.Count;

    public bool IsReadOnly => ((ICollection<ResponsePoint>)Core).IsReadOnly;

    public async Task AddRangeAsync(IAsyncEnumerable<ResponsePoint> values)
    {
        await foreach (ResponsePoint r in values)
            Add(r);
    }
    public void AddRange(IEnumerable<ResponsePoint> values)
    {
        foreach (ResponsePoint rp in values)
            Add(rp);
    }
    public void Add(ResponsePoint rp)
    {
        int ind = Core.FindIndex(i => i.Frequency > rp.Frequency);
        if (ind == -1)
            Core.Add(rp);
        else
            Core.Insert(ind, rp);
    }

    public void Clear()
    {
        Core.Clear();
        IsSorted = true;
    }

    public bool Contains(ResponsePoint item) => Core.Contains(item);

    public void CopyTo(ResponsePoint[] array, int arrayIndex) => Core.CopyTo(array, arrayIndex);

    public IEnumerator<ResponsePoint> GetEnumerator() => Core.GetEnumerator();

    public int IndexOf(ResponsePoint item)
    {
        return ((IList<ResponsePoint>)Core).IndexOf(item);
    }

    public void Insert(int index, ResponsePoint item)
    {
        ((IList<ResponsePoint>)Core).Insert(index, item);
    }

    public bool Remove(ResponsePoint item)
    {
        return ((ICollection<ResponsePoint>)Core).Remove(item);
    }

    public void RemoveAt(int index)
    {
        ((IList<ResponsePoint>)Core).RemoveAt(index);
    }

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => ((System.Collections.IEnumerable)Core).GetEnumerator();
}
