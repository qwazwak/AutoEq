using System;
using System.Collections.Generic;

namespace AutoEQ.Helper;

public static class ICollectionExtensions
{
    public static void AddIfNotNull<T>(this ICollection<T> collection, T? val) where T : struct
    {
        if(val != null)
            collection.Add(val.Value);
    }
}
