public class NeuralNetwork
{
    private double LearnRate;
    private int InputNeurons;
    private int HiddenNeurons;
    private int OutputNeurons;
    private double[] HiddenOutputs;
    private double[] Outputs;
    private double[][] InputToHiddenWeights;
    private double[][] HiddenToOutputWeights;
    private Random rnd;

    public NeuralNetwork(int inputs, int hiddenNeurons, int outputs, double learnRate)
    {
        LearnRate = learnRate;
        InputNeurons = inputs;
        HiddenNeurons = hiddenNeurons;
        OutputNeurons = outputs;

        HiddenOutputs = new double[hiddenNeurons];
        Outputs = new double[outputs];

        InputToHiddenWeights = new double[hiddenNeurons][];
        HiddenToOutputWeights = new double[outputs][];

        rnd = new Random();

        for (int i = 0; i < InputToHiddenWeights.Length; i++)
        {
            InputToHiddenWeights[i] = new double[inputs];
            for (int j = 0; j < InputToHiddenWeights[i].Length; j++)
            {
                InputToHiddenWeights[i][j] = rnd.NextDouble();
            }
        }

        for (int i = 0; i < HiddenToOutputWeights.Length; i++)
        {
            HiddenToOutputWeights[i] = new double[hiddenNeurons];
            for (int j = 0; j < HiddenToOutputWeights[i].Length; j++)
            {
                HiddenToOutputWeights[i][j] = rnd.NextDouble();
            }
        }
    }
    public double[] CalculateOutput(double[] inputs)
    {
        for (int i = 0; i < HiddenNeurons; i++)
        {
            HiddenOutputs[i] = 0;
            for (int j = 0; j < InputNeurons; j++)
            {
                HiddenOutputs[i] += InputToHiddenWeights[i][j] * inputs[j];
            }
            HiddenOutputs[i] = Sigmoid(HiddenOutputs[i]);
        }

        for (int i = 0; i < OutputNeurons; i++)
        {
            Outputs[i] = 0;
            for (int j = 0; j < HiddenNeurons; j++)
            {
                Outputs[i] += HiddenToOutputWeights[i][j] * HiddenOutputs[j];
            }
            Outputs[i] = Sigmoid(Outputs[i]);
        }

        return Outputs;
    }

    public double[] CalculateOutput_Parallel(double[] inputs)
    {
        var options = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };

        Parallel.For(0, HiddenNeurons, options, i =>
        {
            HiddenOutputs[i] = 0;
            for (int j = 0; j < InputNeurons; j++)
            {
                HiddenOutputs[i] += InputToHiddenWeights[i][j] * inputs[j];
            }
            HiddenOutputs[i] = Sigmoid(HiddenOutputs[i]);
        });

        Parallel.For(0, OutputNeurons, options, i =>
        {
            Outputs[i] = 0;
            for (int j = 0; j < HiddenNeurons; j++)
            {
                Outputs[i] += HiddenToOutputWeights[i][j] * HiddenOutputs[j];
            }
            Outputs[i] = Sigmoid(Outputs[i]);
        });

        return Outputs;
    }


    public void Train(double[] inputs, double[] targets)
    {
        double[] outputs = CalculateOutput(inputs);
        double[] outputErrors = new double[outputs.Length];
        double[] hiddenErrors = new double[HiddenNeurons];

        for (int i = 0; i < outputErrors.Length; i++)
        {
            outputErrors[i] = targets[i] - outputs[i];
        }

        for (int i = 0; i < HiddenToOutputWeights.Length; i++)
        {
            for (int j = 0; j < HiddenToOutputWeights[i].Length; j++)
            {
                HiddenToOutputWeights[i][j] += LearnRate * outputErrors[i] * HiddenOutputs[j];
            }
        }

        for (int i = 0; i < hiddenErrors.Length; i++)
        {
            for (int j = 0; j < outputErrors.Length; j++)
            {
                hiddenErrors[i] += HiddenToOutputWeights[j][i] * outputErrors[j];
            }
        }

        for (int i = 0; i < InputToHiddenWeights.Length; i++)
        {
            for (int j = 0; j < InputToHiddenWeights[i].Length; j++)
            {
                InputToHiddenWeights[i][j] += LearnRate * hiddenErrors[i] * inputs[j];
            }
        }
    }


    public void Train_Parallel(double[] inputs, double[] targets)
    {
        double[] outputs = CalculateOutput(inputs);
        double[] outputErrors = new double[outputs.Length];
        double[] hiddenErrors = new double[HiddenNeurons];

        for (int i = 0; i < outputErrors.Length; i++)
        {
            outputErrors[i] = targets[i] - outputs[i];
        }
        var options = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };
        Parallel.For(0, HiddenToOutputWeights.Length, options, i =>
        {
            for (int j = 0; j < HiddenToOutputWeights[i].Length; j++)
            {
                HiddenToOutputWeights[i][j] += LearnRate * outputErrors[i] * HiddenOutputs[j];
            }
        });

        for (int i = 0; i < hiddenErrors.Length; i++)
        {
            for (int j = 0; j < outputErrors.Length; j++)
            {
                hiddenErrors[i] += HiddenToOutputWeights[j][i] * outputErrors[j];
            }
        }

        Parallel.For(0, InputToHiddenWeights.Length, options, i =>
        {
            for (int j = 0; j < InputToHiddenWeights[i].Length; j++)
            {
                InputToHiddenWeights[i][j] += LearnRate * hiddenErrors[i] * inputs[j];
            }
        });
    }
    public double Sigmoid(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }
}

