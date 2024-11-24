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

            //Normalization has exclusive bounds
            Perceptron gradientDescent = new Perceptron(3, 0.1, MeanSquared, TanH, 0, 10, 0, 1);
            gradientDescent.Randomize(new Random(), 0, 1);
            double[][] inputs = [[1], [5], [9]];
            double[] desiredOutputs = [9, 1, 9];

            double error = gradientDescent.Train(inputs, desiredOutputs);
            double originalError = error;
            while (true)
            {
                Console.WriteLine("Starting Error:");
                Console.WriteLine(originalError);
                Console.WriteLine("      ");

                Console.WriteLine("Current Data:");
                error = gradientDescent.Train(inputs, desiredOutputs);
                Console.Write(error);

                Console.Write("      ");
                var output = gradientDescent.Compute(inputs);
                for (int i = 0; i < output.Length; i++) Console.Write(output[i] + " ");
                Console.ReadKey();
                Console.Clear();
            }
        }
    }
}