using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Optimization;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AutoEQ.ConsoleUI.Development;
public class Class1
{
    private static readonly Func<Vector<double>, Vector<double>, Vector<double>> FuncWrap = (v1, v2) => new DenseVector(new double[] { Func(v1[0], v2[0]) });
    private static readonly Func<double, double, double> Func = (x, y) => Math.Pow(x, 2) / Math.Pow(y, x);
    public static void Main()
    {
        LevenbergMarquardtMinimizer Opt = new()
        {
                
        };

        Vector Guess = new DenseVector(new double[] { 1, 8 });
        //var objective = (IObjectiveModel)ObjectiveFunction.NonlinearFunction(FuncWrap, new DenseVector(new double[1]), new DenseVector(new double[1]));
        var objective = new NonlinearObjectiveFunction(FuncWrap);
        double[] X = Enumerable.Range(0, 10).Select(_ => Random.Shared.NextDouble() * 100).ToArray();
        double[] Y = X.Select(x => Random.Shared.NextDouble() * 100).ToArray();
        objective.SetObserved(new DenseVector(new double[1]), new DenseVector(new double[1]));
        var res = Opt.FindMinimum(objective, Guess);
        Console.WriteLine(res.ReasonForExit);
        Console.WriteLine(res.MinimizedValues.ToString());
        Console.WriteLine(res.MinimizingPoint.ToString());
    }
    
    /*public static void MainSimple()
    {
        NelderMeadSimplex Opt = new(.0005, 10000)
        {
                
        };

        Vector Guess = new DenseVector(new double[] { 1, 8 });
        var objective = ObjectiveFunction.Value(FuncWrap);
        MinimizationResult res = Opt.FindMinimum(objective, Guess);
        Console.WriteLine(res.ReasonForExit);
        Console.WriteLine(res.FunctionInfoAtMinimum.ToString());
        Console.WriteLine(res.MinimizingPoint.ToString());
        Console.WriteLine(FuncWrap(res.MinimizingPoint));
    }*/
    internal class ValueObjectiveModel : IObjectiveModel, IObjectiveFunction
    {
        private readonly Func<MathNet.Numerics.LinearAlgebra.Vector<double>, double> _function;

        public bool IsGradientSupported => false;

        public bool IsHessianSupported => false;

        public MathNet.Numerics.LinearAlgebra.Vector<double> Point
        {
            get;
            private set;
        }

        public double Value
        {
            get;
            private set;
        }

        public MathNet.Numerics.LinearAlgebra.Matrix<double> Hessian
        {
            get
            {
                throw new NotSupportedException();
            }
        }

        public MathNet.Numerics.LinearAlgebra.Vector<double> Gradient
        {
            get
            {
                throw new NotSupportedException();
            }
        }

        public Vector<double> ObservedY => throw new NotImplementedException();

        public Matrix<double> Weights => throw new NotImplementedException();

        public Vector<double> ModelValues => throw new NotImplementedException();

        public int FunctionEvaluations { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
        public int JacobianEvaluations { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        public int DegreeOfFreedom => throw new NotImplementedException();

        public ValueObjectiveModel(Func<MathNet.Numerics.LinearAlgebra.Vector<double>, double> function)
        {
            _function = function;
        }

        public IObjectiveModel CreateNew()
        {
            return new ValueObjectiveModel(_function);
        }

        public IObjectiveModel Fork()
        {
            return new ValueObjectiveModel(_function)
            {
                Point = Point,
                Value = Value
            };
        }

        public void EvaluateAt(MathNet.Numerics.LinearAlgebra.Vector<double> point)
        {
            Point = point;
            Value = _function(point);
        }

        IObjectiveFunction IObjectiveFunction.Fork()
        {

            return new ValueObjectiveModel(_function)
            {
                Point = Point,
                Value = Value
            };
        }

        IObjectiveFunction IObjectiveFunctionEvaluation.CreateNew()
        {
            return new ValueObjectiveModel(_function);
        }

        public void SetParameters(Vector<double> initialGuess, List<bool> isFixed = null)
        {
            throw new NotImplementedException();
        }

        public IObjectiveFunction ToObjectiveFunction()
        {
            return this;
        }
    }
}