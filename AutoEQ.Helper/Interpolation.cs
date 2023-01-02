using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AutoEQ.Helper;
public class Interpolation
{
    public IList<double> xs { get; }
    public IList<double> ys { get; }
    public int count { get; }

    public Interpolation(IEnumerable<double> xs, IEnumerable<double> ys, int count) : this(xs as IList<double> ?? xs.ToList(), ys as IList<double> ?? ys.ToList(), count)
    {
    }
    public Interpolation(IList<double> xs, IList<double> ys, int count)
    {
        this.xs = xs;
        this.ys = ys;
        this.count = count;
        LazyResult = new(() => Interpolate1D(this.xs, this.ys, this.count));
    }
    private Lazy<(List<double> Result_xs, List<double> Result_ys)> LazyResult;
    public List<double> Result_xs => LazyResult.Value.Result_xs;
    public List<double> Result_ys => LazyResult.Value.Result_ys;

    public double Interpolate(double X)
    {
        int Index = Result_xs.IndexOf(X);
        if (Index != -1)
            return Result_ys[Index];
        int Next = Result_xs.FindIndex(v => v > X);
        int Prev = Next > 1 ? Next - 1 : 0;
        double T = UnLerp(Result_xs[Prev], Result_xs[Next], X);
        double Y = Lerp(Result_ys[Prev], Result_ys[Next], X);

        Result_xs.Insert(Next, X);
        Result_ys.Insert(Next, Y);
        return Y;
    }
    public IEnumerable<double> Interpolate(IEnumerable<double> Xs)
    {
        foreach (double x in Xs)
            yield return Interpolate(x);
    }

    private static double Lerp(double A, double B, double T) => A + ((B - A) * T);
    private static double UnLerp(double min, double max, double value) => (value - min) / (max - min);

    public static (List<double> xs, List<double> ys) Interpolate1D(IList<double> xs, IList<double> ys, int count)
    {
        if (xs is null || ys is null || xs.Count != ys.Count)
            throw new ArgumentException($"{nameof(xs)} and {nameof(ys)} must have same length");

        int inputPointCount = xs.Count;
        double[] inputDistances = new double[inputPointCount];
        inputDistances[0] = 0;
        for (int i = 1; i < inputPointCount; i++)
            inputDistances[i] = inputDistances[i - 1] + xs[i] - xs[i - 1];

        double meanDistance = inputDistances.Last() / (count - 1);
        double[] evenDistances = Enumerable.Range(0, count).Select(x => x * meanDistance).ToArray();

        List<double> xsOut = null!;
        List<double> ysOut = null!;
        Parallel.Invoke(
            () => xsOut = Interpolate(inputDistances, xs, evenDistances),
            () => ysOut = Interpolate(inputDistances, ys, evenDistances));
        return (xsOut, ysOut);
    }

    private static List<double> Interpolate(IList<double> xOrig, IList<double> yOrig, IList<double> xInterp)
    {
        (IList<double> a, IList<double> b) = FitMatrix(xOrig, yOrig);

        //double[] yInterp = new double[xInterp.Count];
        List<double> yInterp = new(xInterp.Count);
        for (int i = 0; i < yInterp.Count; i++)
        {
            int j;
            for (j = 0; j < xOrig.Count - 2; j++)
                if (xInterp[i] <= xOrig[j + 1])
                    break;

            double dx = xOrig[j + 1] - xOrig[j];
            double t = (xInterp[i] - xOrig[j]) / dx;
            double y = ((1 - t) * yOrig[j]) + (t * yOrig[j + 1]) +
                   (t * (1 - t) * ((a[j] * (1 - t)) + (b[j] * t)));
            yInterp[i] = y;
        }

        return yInterp;
    }

    private static (IList<double> a, IList<double> b) FitMatrix(IList<double> x, IList<double> y)
    {
        int n = x.Count;
        double[] r = new double[n];
        double[] A = new double[n];
        double[] B = new double[n];
        double[] C = new double[n];

        double dx1, dx2, dy1, dy2;

        dx1 = x[1] - x[0];
        C[0] = 1.0f / dx1;
        B[0] = 2.0f * C[0];
        r[0] = 3 * (y[1] - y[0]) / (dx1 * dx1);

        for (int i = 1; i < n - 1; i++)
        {
            dx1 = x[i] - x[i - 1];
            dx2 = x[i + 1] - x[i];
            A[i] = 1.0f / dx1;
            C[i] = 1.0f / dx2;
            B[i] = 2.0f * (A[i] + C[i]);
            dy1 = y[i] - y[i - 1];
            dy2 = y[i + 1] - y[i];
            r[i] = 3 * ((dy1 / (dx1 * dx1)) + (dy2 / (dx2 * dx2)));
        }

        dx1 = x[n - 1] - x[n - 2];
        dy1 = y[n - 1] - y[n - 2];
        A[n - 1] = 1.0f / dx1;
        B[n - 1] = 2.0f * A[n - 1];
        r[n - 1] = 3 * (dy1 / (dx1 * dx1));

        double[] cPrime = new double[n];
        cPrime[0] = C[0] / B[0];
        for (int i = 1; i < n; i++)
            cPrime[i] = C[i] / (B[i] - (cPrime[i - 1] * A[i]));

        double[] dPrime = new double[n];
        dPrime[0] = r[0] / B[0];
        for (int i = 1; i < n; i++)
            dPrime[i] = (r[i] - (dPrime[i - 1] * A[i])) / (B[i] - (cPrime[i - 1] * A[i]));

        double[] k = new double[n];
        k[n - 1] = dPrime[n - 1];
        for (int i = n - 2; i >= 0; i--)
            k[i] = dPrime[i] - (cPrime[i] * k[i + 1]);

        IList<double> a = new List<double>(n - 1);
        IList<double> b = new List<double>(n - 1);
        //double[] b = new double[n - 1];
        for (int i = 1; i < n; i++)
        {
            dx1 = x[i] - x[i - 1];
            dy1 = y[i] - y[i - 1];
            a[i - 1] = (k[i - 1] * dx1) - dy1;
            b[i - 1] = (-k[i] * dx1) + dy1;
        }
        return (a, b);
    }
}
