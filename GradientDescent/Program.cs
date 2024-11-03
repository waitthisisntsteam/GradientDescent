using System.Drawing;
using System.Reflection.PortableExecutable;
using System.Transactions;

namespace GradientDescent
{
    internal class Program
    {
        static void Main(string[] args)
        {
            ActivationErorrFormulas activationErorrFormulas = new ActivationErorrFormulas();
            ErrorFunction MeanSquared = new ErrorFunction(activationErorrFormulas.MeanSquared, activationErorrFormulas.MeanSquaredD);
            ActivationFunction TanH = new ActivationFunction(activationErorrFormulas.TanH, activationErorrFormulas.TanHD);

            Perceptron gradientDescent = new Perceptron(3, 0.5, MeanSquared, TanH);
            gradientDescent.Randomize(new Random(), 100, 300);
            double[][] inputs = [[100], [200], [300]];
            double[] desiredOutputs = [300, 100, 300];

            double error = gradientDescent.Train(inputs, desiredOutputs);
            Console.WriteLine(error);
        }
    }
}