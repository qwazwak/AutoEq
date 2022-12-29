using System.Collections.Generic;
using System.Linq;

namespace AutoEQ2.Helper;

public static class MathEx
{
    public static double DotProduct(this IEnumerable<double> A, IEnumerable<double> B) => A.MultiplyPairs(B).Sum();
    public static IEnumerable<double> MultiplyPairs(this IEnumerable<double> A, IEnumerable<double> B) => A.Zip(B, (a, b) => a * b);
}
