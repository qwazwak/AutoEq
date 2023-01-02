using System;
using System.Collections.Generic;
using System.Linq;

namespace AutoEQ.Helper;

public class IReadonlyListWrapper<TContainer, T> : IReadOnlyList<T>
{
    protected readonly IList<TContainer> ListReference;
    protected readonly Func<TContainer, T> Unwrap;

    public IReadonlyListWrapper(IList<TContainer> ListReference, Func<TContainer, T> Unwrap)
    {
        this.ListReference = ListReference;
        this.Unwrap = Unwrap;
    }

    public T this[int index] => Unwrap.Invoke(ListReference[index]);

    public int Count => ListReference.Count;

    public IEnumerator<T> GetEnumerator() => ListReference.Select(Unwrap).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
