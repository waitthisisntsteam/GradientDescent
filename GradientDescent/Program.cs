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

            Perceptron gradientDescent = new Perceptron(3, 0.1, MeanSquared, TanH);
            gradientDescent.Randomize(new Random(), 0, 1);
            double[][] inputs = [[1], [2], [3]];
            double[] desiredOutputs = [1, 0, 1];

            double error = 0;
            while (true)
            {
                error = gradientDescent.Train(inputs, desiredOutputs);
                Console.WriteLine(error);
            }
        }
    }
}