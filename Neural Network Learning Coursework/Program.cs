using Neural_Network_Learning_Coursework;
using System;
using System.Diagnostics;
using OfficeOpenXml;

static class Program
{
    static void Main(string[] args)
    {
        int hiddenNeurons = 200 ;
        int outputs = 50 ;
        double learnRate = 0.1;
        int epochs = 10;
        int testSetSize = 1000;

        // Розміри вхідних даних, які будемо використовувати для експериментів
        int[] inputSizes = new int[] { 100, 200, 500, 1000, 2000, 5000 };

        foreach (var inputs in inputSizes)
        {
            Console.WriteLine($"Running tests with input size: {inputs}");
            NeuralNetworkTest nnTest = new NeuralNetworkTest(inputs, hiddenNeurons, outputs, learnRate);
            nnTest.RunTests(inputs, testSetSize, epochs);
        }
    }
}

