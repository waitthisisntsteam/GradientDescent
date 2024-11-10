﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GradientDescent
{
    public class ActivationErorrFormulas
    {
        // Activation Functions
        public double Sigmoid(double input) => 1 / (1 + Math.Pow(double.E, -input));
        public double SigmoidD(double input) => Sigmoid(input) * (1 - Sigmoid(input));

        public double TanH(double input) //=> (Math.Pow(double.E, input) - Math.Pow(double.E, -input)) / (Math.Pow(double.E, input) + Math.Pow(double.E, -input));
        {
            //double numerator = Math.Pow(double.E, input) - Math.Pow(double.E, -input);
            //double denominator = Math.Pow(double.E, input) + Math.Pow(double.E, -input);
            return Math.Tanh(input);
            //return numerator / denominator;
        }

        public double TanHD(double input) //=> 1 - Math.Pow(TanH(input), 2);
        {
            double fOfX = TanH(input);
            double fofXSquared = Math.Pow(fOfX, 2);
            double output = 1 - fofXSquared;

            return output;
        }

        // Error Functions
        public double MeanSquared(double input, double desiredOutput) => Math.Pow(desiredOutput - input, 2);
        public double MeanSquaredD(double input, double desiredOutput) => -2 * (desiredOutput - input);
    }
}
