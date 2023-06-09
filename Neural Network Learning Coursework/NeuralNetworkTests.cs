using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network_Learning_Coursework
{
    public class NeuralNetworkTest
    {
        private readonly NeuralNetwork _nn;
        private readonly int _inputs;
        private readonly int _outputs;

        public NeuralNetworkTest(int inputs, int hiddenNeurons, int outputs, double learnRate)
        {
            _inputs = inputs;
            _outputs = outputs;
            _nn = new NeuralNetwork(inputs, hiddenNeurons, outputs, learnRate);
        }

        public void RunTests(int trainSetSize, int testSetSize, int epochs)
        {
            Random random = new Random();

            // Generate random training inputs and targets
            var trainInputs = GenerateRandomData(trainSetSize, _inputs, random);
            var trainTargets = GenerateRandomData(trainSetSize, _outputs, random);

            //var trainTargets = GenerateRandomData(trainSetSize, _outputs, random);

            // Generate random testing inputs and targets
            var testInputs = GenerateRandomData(testSetSize, _inputs, random);
            var testTargets = GenerateRandomData(testSetSize, _outputs, random);

            // Test the training speed of the neural network
            TestTrainingSpeed(trainInputs, trainTargets, epochs);

            // Test the accuracy of the neural network
            TestAccuracy(testInputs, testTargets);

            // Test the prediction speed of the neural network
            TestPredictionSpeed(testInputs);
        }

        private void TestAccuracy(double[][] testInputs, double[][] testTargets)
        {
            int correct = 0;
            for (int i = 0; i < testInputs.Length; i++)
            {
                var prediction = _nn.CalculateOutput(testInputs[i]);
                if (prediction.ToList().IndexOf(prediction.Max()) == testTargets[i].ToList().IndexOf(testTargets[i].Max()))
                    correct++;
            }

            double accuracy = (double)correct / testInputs.Length * 100;
            Console.WriteLine($"Accuracy: {accuracy}%");
        }


        private void TestPredictionSpeed(double[][] testInputs)
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            foreach (var input in testInputs)
            {
                _nn.CalculateOutput(input);
            }

            stopwatch.Stop();
            Console.WriteLine($"Prediction time: {stopwatch.ElapsedMilliseconds} ms");
        }


        private double[][] GenerateRandomData(int setSize, int dataLength, Random random)
        {
            double[][] data = new double[setSize][];
            for (int i = 0; i < setSize; i++)
            {
                data[i] = new double[dataLength];
                for (int j = 0; j < dataLength; j++)
                {
                    data[i][j] = random.NextDouble();
                }
            }

            return data;
        }
        private void TestTrainingSpeed(double[][] trainInputs, double[][] trainTargets, int epochs)
        {
            // Sequential training
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            for (int i = 0; i < epochs; i++)
            {
                for (int j = 0; j < trainInputs.Length; j++)
                {
                    _nn.Train(trainInputs[j], trainTargets[j]);
                }
            }

            stopwatch.Stop();
            Console.WriteLine($"Training time (sequential): {stopwatch.ElapsedMilliseconds} ms");

            // Parallel training
            Stopwatch stopwatchParallel = new Stopwatch();
            stopwatchParallel.Start();

            for (int i = 0; i < epochs; i++)
            {
                Parallel.For(0, trainInputs.Length, j =>
                {
                    _nn.Train_Parallel(trainInputs[j], trainTargets[j]);
                });
            }

            stopwatchParallel.Stop();
            Console.WriteLine($"Training time (parallel): {stopwatchParallel.ElapsedMilliseconds} ms");
        }
    }

}
