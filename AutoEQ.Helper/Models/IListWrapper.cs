using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace AutoEQ.Helper;

public class ICollectionWrapper<TContainer, T> : ICollection<T> where T : IEquatable<T>
{
    protected readonly ICollection<TContainer> BaseCollection;
    protected readonly Func<TContainer, T> Unwrap;
    protected readonly Action<T, TContainer> SetWrap;

    public ICollectionWrapper(IList<TContainer> BaseCollection, Func<TContainer, T> Unwrap, Action<T, TContainer> SetWrap)
    {
        this.BaseCollection = BaseCollection;
        this.Unwrap = Unwrap;
        this.SetWrap = SetWrap;
    }

    public int Count => BaseCollection.Count;

    public bool IsReadOnly => BaseCollection.IsReadOnly;

    public void Add(T item) => throw new NotSupportedException();

    public void Clear() => BaseCollection.Clear();

    public bool Contains(T item) => BaseCollection.Any(c => item.Equals(Unwrap.Invoke(c)));

    public void CopyTo(T[] array, int arrayIndex)
    {
        using IEnumerator<T> e = GetEnumerator();
        for (int i = arrayIndex; i < array.Length; i++)
        {
            if (e.MoveNext())
                array[i] = e.Current;
            else
                break;
        }
        if (e.MoveNext())
            throw new ArgumentException("Destination array was not long enough. Check the destination index, length, and the array's lower bounds.");
    }

    public IEnumerator<T> GetEnumerator() => BaseCollection.Select(Unwrap).GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    public bool Remove(T item) => throw new NotSupportedException();
}

public class IListWrapper<TContainer, T> : ICollectionWrapper<TContainer, T>, IList<T> where T : IEquatable<T>
{
    protected readonly IList<TContainer> BaseList;

    public IListWrapper(IList<TContainer> BaseList, Func<TContainer, T> Unwrap, Action<T, TContainer> SetWrap) : base(BaseList, Unwrap, SetWrap)
    {
        this.BaseList = BaseList;
    }

    public T this[int index] { get => Unwrap.Invoke(BaseList[index]); set => SetWrap.Invoke(value, BaseList[index]); }

    public int IndexOf(T item)
    {
        for (int i = 0; i < BaseList.Count; i++)
        {
            TContainer container = BaseList[i];
            if (item.Equals(Unwrap.Invoke(container)))
                return i;
        }
        return -1;
    }

    public void Insert(int index, T item) => throw new NotSupportedException();

    public void RemoveAt(int index) => BaseList.RemoveAt(index);
}
