using System;
using System.Threading;

namespace AutoEQ.Helper;

public class ResettableLazy<T>
{
    private readonly Func<T> ValueFactory;
    private Lazy<T> Core;

    public bool IsValueCreated => Core.IsValueCreated;

    public T Value => Core.Value;

    public LazyThreadSafetyMode LazyThreadSafetyMode { get; }

    public ResettableLazy(Func<T> ValueFactory, LazyThreadSafetyMode LazyThreadSafetyMode = LazyThreadSafetyMode.ExecutionAndPublication)
    {
        this.ValueFactory = ValueFactory;
        this.LazyThreadSafetyMode = LazyThreadSafetyMode;
        Core = new(this.ValueFactory, this.LazyThreadSafetyMode);
    }

    public ResettableLazy(Func<T> valueFactory, bool isThreadSafe) : this(valueFactory, isThreadSafe ? LazyThreadSafetyMode.ExecutionAndPublication : LazyThreadSafetyMode.None)
    {
    }

    public void Reset() => Core = new(ValueFactory, LazyThreadSafetyMode);
}
