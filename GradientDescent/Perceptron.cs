using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientDescent
{
    class Perceptron
    {
        public double LearningRate { get; set; }

        double[] Weights;
        double Bias;

        ActivationFunction ActivationFunction;
        ErrorFunction ErrorFunction;

        public Perceptron(int inputAmount, double learningRate, ErrorFunction errorFunction, ActivationFunction activationFunction)
        {
            Weights = new double[inputAmount];
            Bias = 0;

            LearningRate = learningRate;

            ActivationFunction = activationFunction;
            ErrorFunction = errorFunction;
        }

        private double Random(Random random, double min, double max) => (random.NextDouble() * (max - min)) + min;
        public void Randomize(Random random, double min, double max)
        {
            for (int i = 0; i < Weights.Length; i++) Weights[i] = Random(random, min, max);
            Bias = Random(random, min, max);
        }

        public double[] UserComputeWithFiltering(double[][] inputs)
        {
            double[] outputs = Compute(inputs);

            for (int i = 0; i < outputs.Length; i++)
            {
                if (outputs[i] < 0.5) outputs[i] = 0;
                else outputs[i] = 1;
            }

            return outputs;
        }

        public double[] UserComputeWithActivation(double[][] inputs)
        {
            double[] outputs = Compute(inputs);

            for (int i = 0; i < outputs.Length; i++) outputs[i] = ActivationFunction.FunctionFunc(outputs[i]);

            return outputs;
        }

        private double[] Compute(double[][] inputs)
        {
            double[] output = new double[inputs.Length];
            for (int i = 0; i < inputs.Length; i++) output[i] = Compute(inputs[i]);
            return output;
        }

        private double Compute(double[] inputs)
        {
            double output = Bias;
            for (int i = 0; i < inputs.Length; i++) output += inputs[i] * Weights[i];
            return output;
        }

        public double GetError(double[][] inputs, double[] desiredOutputs)
        {
            double[] outputs = Compute(inputs);

            double errorSum = 0;
            for (int i = 0; i < outputs.Length; i++) errorSum += Math.Pow(ErrorFunction.FunctionFunc(outputs[i], desiredOutputs[i]), 2);
            return errorSum / outputs.Length;
        }

        public double Train(double[][] inputs, double[] desiredOutputs)
        {

            //ADD NORMALIZING

            Random rand = new Random();
            int chosenIndex = rand.Next(0, Weights.Length + 1);

            int i = 0; //chosen weight
            int z = 0; //sum of all weight inputs
            double actualOutput = Compute(inputs)[chosenIndex];
            double weightPartialDerivative = ErrorFunction.DerivativeFunc(actualOutput, desiredOutputs[i]) * ActivationFunction.DerivativeFunc(Bias + z) * inputs[i][chosenIndex];
            double biasPartialDerivative = ErrorFunction.DerivativeFunc(actualOutput, desiredOutputs[i]) * ActivationFunction.DerivativeFunc(Bias + z);

            if (chosenIndex < Weights.Length) Weights[chosenIndex] += LearningRate * -weightPartialDerivative;
            else Bias += LearningRate * -biasPartialDerivative;

            return GetError(inputs, desiredOutputs);
        }
    }
}
