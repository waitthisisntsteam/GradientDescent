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

        private double Min;
        private double Max;
        private double NMin;
        private double NMax;

        public Perceptron(int inputAmount, double learningRate, ErrorFunction errorFunction, ActivationFunction activationFunction, double min, double max, double nMin, double nMax)
        {
            Weights = new double[inputAmount];
            Bias = 0;

            LearningRate = learningRate;

            ActivationFunction = activationFunction;
            ErrorFunction = errorFunction;

            Min = min;
            Max = max;
            NMin = nMin;
            NMax = nMax;
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

        public double[] Compute(double[][] inputs)
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

        public double Normalize(double num) => (num - Min) / (Max - Min) * (NMax - NMin) + NMin;
        public double Unnormalize(double num) => (num - NMin) / (NMax - NMin) * (Max - Min) + Min;
        public double Train(double[][] inputs, double[] desiredOutputs)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                for (int j = 0; j < inputs[i].Length; j++) inputs[i][j] = Normalize(inputs[i][j]);
            }
            for (int i = 0; i < desiredOutputs.Length; i++) desiredOutputs[i] = Normalize(desiredOutputs[i]);

            double[] weightPartialDerivatives = new double[Weights.Length];
            double[] actualOutputs = Compute(inputs);
            double sumOfAllWeightInputs = 0;
            for (int i = 0; i < actualOutputs.Length; i++) sumOfAllWeightInputs += actualOutputs[i];
            
            for (int i = 0; i < Weights.Length; i++)
            {
                for (int j = 0; j < inputs.Length; j++)
                {
                    double actualOutput = actualOutputs[j];
                    double ePrimeOD = ErrorFunction.DerivativeFunc(ActivationFunction.FunctionFunc(actualOutput), desiredOutputs[j]);
                    double aPrimeZ = ActivationFunction.DerivativeFunc(Bias + sumOfAllWeightInputs);
                    double xI = inputs[j][0];

                    double weightPartialDerivative = ePrimeOD * aPrimeZ * xI;
                    weightPartialDerivatives[i] += weightPartialDerivative;
                }
            }
            double actualOutputB = actualOutputs[0];
            double ePrimeODB = ErrorFunction.DerivativeFunc(actualOutputB, desiredOutputs[0]);
            double aPrimeZB = ActivationFunction.DerivativeFunc(Bias + sumOfAllWeightInputs);

            double biasPartialDerivative = ePrimeODB * aPrimeZB;

            Bias += LearningRate * -biasPartialDerivative;
            for (int i = 0; i < Weights.Length; i++) Weights[i] += LearningRate * -weightPartialDerivatives[i];

            for (int i = 0; i < inputs.Length; i++)
            {
                for (int j = 0; j < inputs[i].Length; j++) inputs[i][j] = Unnormalize(inputs[i][j]);
            }
            for (int i = 0; i < desiredOutputs.Length; i++) desiredOutputs[i] = Unnormalize(desiredOutputs[i]);

            return GetError(inputs, desiredOutputs);
        }
    }
}
