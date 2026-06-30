using System;
using System.IO;
using System.Collections.Generic;

namespace NeuralNetworkDeepRegression
{
  internal class NeuralNetworkDeepRegressionProgram
  {
    static void Main(string[] args)
    {
      Console.WriteLine("\nDeep neural network regression using C# ");

      // 1. load data
      Console.WriteLine("\nLoading synthetic train (200) and test (40) data ");
      string trainFile = "C:\\VSR\\Data\\synthetic_train_200.txt";
      int[] colsX = new int[] { 0, 1, 2, 3, 4 };
      double[][] trainX = MatLoad(trainFile, colsX, ',', "#");
      double[] trainY = MatToVec(MatLoad(trainFile, new int[] { 5 },
        ',', "#"));

      string testFile = "C:\\VSR\\Data\\synthetic_test_40.txt";
      double[][] testX = MatLoad(testFile, colsX, ',', "#");
      double[] testY = MatToVec(MatLoad(testFile, new int[] { 5 },
        ',', "#"));
      Console.WriteLine("Done ");

      Console.WriteLine("\nFirst three train X: ");
      for (int i = 0; i < 3; ++i)
        VecShow(trainX[i], 4, 8);

      Console.WriteLine("\nFirst three train y: ");
      for (int i = 0; i < 3; ++i)
        Console.WriteLine(trainY[i].ToString("F4").
          PadLeft(8));

      // 2. create NN
      Console.WriteLine("\nCreating 5-20-20-1 tanh()" +
        " identity() neural network regressor ");
      NeuralNetworkDeepRegressor nn =
         new NeuralNetworkDeepRegressor(5, 20, 20, 1);
      Console.WriteLine("Done ");

      // 3. train NN
      double lrnRate = 0.04;
      int maxEpochs = 5000;
      double decay = 0.000001;

      Console.WriteLine("\nSetting lrnRate = " + lrnRate.ToString("F4"));
      Console.WriteLine("Setting maxEpochs = " + maxEpochs);
      Console.WriteLine("Setting decay = " + decay.ToString("F8"));
      Console.WriteLine("\nStarting training ");
      nn.Train(trainX, trainY, lrnRate, maxEpochs, decay);
      Console.WriteLine("Done ");

      // TODO: programmatically analyze weights and biases for extreme values

      // 4. evaluate trained model
      Console.WriteLine("\nEvaluating model ");
      double trainAcc = nn.Accuracy(trainX, trainY, 0.05);
      double testAcc = nn.Accuracy(testX, testY, 0.05);
      Console.WriteLine("\nAccuracy (5%) on train data = " +
        trainAcc.ToString("F4"));
      Console.WriteLine("Accuracy (5%) on test data  = " +
        testAcc.ToString("F4"));

      double trainMSE = nn.MSE(trainX, trainY);
      double testMSE = nn.MSE(testX, testY);
      Console.WriteLine("\nMSE on train data = " +
        trainMSE.ToString("F4"));
      Console.WriteLine("MSE on test data = " +
        testMSE.ToString("F4"));

      // 5. use model
      Console.WriteLine("\nPredicting y for train[0] ");
      double[] x = trainX[0];
      double predY = nn.Predict(x);
      Console.WriteLine("Predicted y = " + predY.ToString("F4"));

      Console.WriteLine("\nEnd demo ");
      Console.ReadLine();

    } // Main

    // ------------------------------------------------------
    // helpers for Main(): MatLoad(), MatToVec(), VecShow()
    // ------------------------------------------------------

    static double[][] MatLoad(string fn, int[] usecols,
      char sep, string comment)
    {
      List<double[]> result = new List<double[]>();
      string line = "";
      FileStream ifs = new FileStream(fn, FileMode.Open);
      StreamReader sr = new StreamReader(ifs);
      while ((line = sr.ReadLine()) != null)
      {
        if (line.StartsWith(comment) == true)
          continue;
        string[] tokens = line.Split(sep);
        List<double> lst = new List<double>();
        for (int j = 0; j < usecols.Length; ++j)
          lst.Add(double.Parse(tokens[usecols[j]]));
        double[] row = lst.ToArray();
        result.Add(row);
      }
      sr.Close(); ifs.Close();
      return result.ToArray();
    }

    static double[] MatToVec(double[][] M)
    {
      int nRows = M.Length;
      int nCols = M[0].Length;
      double[] result = new double[nRows * nCols];
      int k = 0;
      for (int i = 0; i < nRows; ++i)
        for (int j = 0; j < nCols; ++j)
          result[k++] = M[i][j];
      return result;
    }

    static void VecShow(double[] vec, int dec, int wid)
    {
      for (int i = 0; i < vec.Length; ++i)
        Console.Write(vec[i].ToString("F" + dec).
          PadLeft(wid));
      Console.WriteLine("");
    }

    // ------------------------------------------------------------------------

  } // class Program

  // ==========================================================================

  public class NeuralNetworkDeepRegressor
  {
    // two hidden layers
    public int numInput;
    public int numHiddenA;
    public int numHiddenB;
    public int numOutput;

    public double[] iNodes;  // input nodes
    public double[] aNodes;
    public double[] bNodes;
    public double[] oNodes;  // output nodes

    public double[][] iaWeights; // input to hidden A
    public double[][] abWeights; // hidden A to hidden B
    public double[][] boWeights; // hidden B to output

    public double[] aBiases;
    public double[] bBiases;
    public double[] oBiases;

    private Random rnd;

    // ------------------------------------------------------------------------

    public NeuralNetworkDeepRegressor(int numInput,
      int numHiddenA, int numHiddenB, int numOutput,
      int seed = 0)
    {
      this.numInput = numInput;
      this.numHiddenA = numHiddenA;
      this.numHiddenB = numHiddenB;
      this.numOutput = numOutput; // always 1 for regression

      this.iNodes = new double[numInput];
      this.iaWeights = MatMake(numInput, numHiddenA);
      this.aBiases = new double[numHiddenA];
      this.aNodes = new double[numHiddenA];

      this.abWeights = MatMake(numHiddenA, numHiddenB);
      this.bBiases = new double[numHiddenB];
      this.bNodes = new double[numHiddenB];

      this.boWeights = MatMake(numHiddenB, numOutput);
      this.oBiases = new double[numOutput];
      this.oNodes = new double[numOutput];

      this.rnd = new Random(seed);
    }

    // ------------------------------------------------------------------------

    private static double[][] MatMake(int nRows, int nCols)
    {
      double[][] result = new double[nRows][];
      for (int i = 0; i < nRows; ++i)
        result[i] = new double[nCols];
      return result;
    }

    // ------------------------------------------------------------------------

    public double Predict(double[] x)
    {
      // copy input into iNodes
      for (int i = 0; i < this.numInput; ++i)
        this.iNodes[i] = x[i];

      // compute hidden layer A
      for (int j = 0; j < numHiddenA; ++j)
      {
        double sum = 0.0;
        for (int i = 0; i < this.numInput; ++i)
          sum += this.iNodes[i] * this.iaWeights[i][j];
        sum += this.aBiases[j];
        this.aNodes[j] = HyperTan(sum);
      }

      // compute hidden layer B
      for (int j = 0; j < this.numHiddenB; ++j)
      {
        double sum = 0.0;
        for (int i = 0; i < this.numHiddenA; ++i)
          sum += this.aNodes[i] * this.abWeights[i][j];
        sum += this.bBiases[j];
        this.bNodes[j] = HyperTan(sum);
      }

      // compute output layer
      for (int j = 0; j < this.numOutput; ++j)
      {
        double sum = 0.0;
        for (int i = 0; i < this.numHiddenB; ++i)
          sum += this.bNodes[i] * this.boWeights[i][j];
        sum += this.oBiases[j];
        this.oNodes[j] = Identity(sum);
      }

      return this.oNodes[0];  // single value
    }

    // ------------------------------------------------------------------------

    private static double HyperTan(double x)
    {
      if (x < -6.0) return -1.0;
      else if (x > 6.0) return 1.0;
      else return Math.Tanh(x);
    }

    // ------------------------------------------------------------------------

    private static double Identity(double x)
    {
      return x;
    }

    // ------------------------------------------------------------------------

    public void Train(double[][] trainX, double[] trainY,
      double lrnRate, int maxEpochs, double decay)
    {
      // init weights
      double lo = -0.01; double hi = 0.01;

      for (int i = 0; i < numInput; ++i)
        for (int j = 0; j < numHiddenA; ++j)
          this.iaWeights[i][j] =
            (hi - lo) * this.rnd.NextDouble() + lo;

      for (int i = 0; i < numHiddenA; ++i)
        for (int j = 0; j < numHiddenB; ++j)
          this.abWeights[i][j] =
            (hi - lo) * this.rnd.NextDouble() + lo;

      for (int i = 0; i < numHiddenB; ++i)
        for (int j = 0; j < numOutput; ++j)
          this.boWeights[i][j] =
            (hi - lo) * this.rnd.NextDouble() + lo;

      // each weight and bias has a gradient
      double[][] boGrads = MatMake(numHiddenB, numOutput);
      double[][] abGrads = MatMake(numHiddenA, numHiddenB);
      double[][] iaGrads = MatMake(numInput, numHiddenA);

      double[] oBiasGrads = new double[numOutput];
      double[] bBiasGrads = new double[numHiddenB];
      double[] aBiasGrads = new double[numHiddenA];

      // each output and hidden node has a 'signal'
      //  which is gradient without associated input
      //  (lower case delta in Wikipedia)
      double[] oSignals = new double[numOutput];
      double[] bSignals = new double[numHiddenB];
      double[] aSignals = new double[numHiddenA];

      int[] indices = new int[trainX.Length];
      for (int i = 0; i < indices.Length; ++i)
        indices[i] = i;

      int freq = (int)(maxEpochs / 10); // progress freq
      for (int epoch = 0; epoch < maxEpochs; ++epoch)
      {
        this.Shuffle(indices);
        for (int ii = 0; ii < trainX.Length; ++ii)
        {
          int idx = indices[ii];
          double[] x = trainX[idx];
          double actualY = trainY[idx];
          double predY = this.Predict(x);

          // output node signals depends on target values
          for (int k = 0; k < this.numOutput; ++k)
          {
            double error = predY - actualY;  // standard
            double derivative = 1.0; // identity activation
            oSignals[k] = error * derivative;
          }

          // signal for B nodes depends on output signals
          for (int j = 0; j < numHiddenB; ++j)
          {
            double derivative =
              (1 + this.bNodes[j]) * (1 - this.bNodes[j]);
            double sum = 0.0;
            for (int k = 0; k < numOutput; ++k)
              sum += oSignals[k] * this.boWeights[j][k];
            bSignals[j] = derivative * sum;
          }

          // signal for A nodes, depends on B signals
          for (int j = 0; j < numHiddenA; ++j)
          {
            double derivative =
              (1 + this.aNodes[j]) * (1 - this.aNodes[j]);
            double sum = 0.0;
            for (int k = 0; k < numHiddenB; ++k)
              sum += bSignals[k] * this.abWeights[j][k];
            aSignals[j] = derivative * sum;
          }

          // at this point, all signals have been computed
          // use signals to calculate gradients left-to-right

          for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHiddenA; ++j)
              iaGrads[i][j] = this.iNodes[i] * aSignals[j];

          for (int i = 0; i < numHiddenA; ++i)
            for (int j = 0; j < numHiddenB; ++j)
              abGrads[i][j] = this.aNodes[i] * bSignals[j];

          for (int i = 0; i < numHiddenB; ++i)
            for (int j = 0; j < numOutput; ++j)
              boGrads[i][j] = this.bNodes[i] * oSignals[j];

          // compute bias gradients
          for (int j = 0; j < numHiddenA; ++j)
            aBiasGrads[j] = 1.0 * aSignals[j];
          for (int j = 0; j < numHiddenB; ++j)
            bBiasGrads[j] = 1.0 * bSignals[j];
          for (int j = 0; j < numOutput; ++j)
            oBiasGrads[j] = 1.0 * oSignals[j];

          // before update, decay all weights
          for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHiddenA; ++j)
              this.iaWeights[i][j] *= (1 - decay);

          for (int i = 0; i < numHiddenA; ++i)
            for (int j = 0; j < numHiddenB; ++j)
              this.abWeights[i][j] *= (1 - decay);

          for (int i = 0; i < numHiddenB; ++i)
            for (int j = 0; j < numOutput; ++j)
              this.boWeights[i][j] *= (1 - decay);

          // use gradients to update all weights

          for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHiddenA; ++j)
              this.iaWeights[i][j] -= iaGrads[i][j] * lrnRate;

          for (int i = 0; i < numHiddenA; ++i)
            for (int j = 0; j < numHiddenB; ++j)
              this.abWeights[i][j] -= abGrads[i][j] * lrnRate;

          for (int i = 0; i < numHiddenB; ++i)
            for (int j = 0; j < numOutput; ++j)
              this.boWeights[i][j] -= boGrads[i][j] * lrnRate;

          // update all biases

          for (int j = 0; j < numHiddenA; ++j)
            this.aBiases[j] -= aBiasGrads[j] * lrnRate;

          for (int j = 0; j < numHiddenB; ++j)
            this.bBiases[j] -= bBiasGrads[j] * lrnRate;

          for (int j = 0; j < numOutput; ++j)
            this.oBiases[j] -= oBiasGrads[j] * lrnRate;

        } // ii each train item

        if (epoch % freq == 0 && epoch < maxEpochs)  // display curr MSE
        {
          double mse = this.MSE(trainX, trainY);
          double acc = this.Accuracy(trainX, trainY, 0.05);

          string s1 = "epoch: " + epoch.ToString().PadLeft(6);
          string s2 = "  MSE = " + mse.ToString("F4");
          string s3 = "  acc = " + acc.ToString("F4");
          Console.WriteLine(s1 + s2 + s3);
        }

      } // epoch

      return;
    }

    // ------------------------------------------------------------------------

    private void Shuffle(int[] indices)
    {
      for (int i = 0; i < indices.Length; ++i)
      {
        int r = this.rnd.Next(i, indices.Length);
        int tmp = indices[r];
        indices[r] = indices[i];
        indices[i] = tmp;
      }
    }

    // ------------------------------------------------------------------------

    public double Accuracy(double[][] dataX, double[] dataY, double pctClose)
    {
      int n = dataX.Length;
      int nCorrect = 0; int nWrong = 0;
      for (int i = 0; i < n; ++i)
      {
        double predY = this.Predict(dataX[i]);
        double actualY = dataY[i];
        if (Math.Abs(predY - actualY) <
          Math.Abs(pctClose * actualY))
          ++nCorrect;
        else
          ++nWrong;
      }
      return (nCorrect * 1.0) / (nCorrect + nWrong);
    }

    // ------------------------------------------------------------------------

    public double MSE(double[][] dataX, double[] dataY)
    {
      int n = dataX.Length;
      double sum = 0.0;
      for (int i = 0; i < n; ++i)
      {
        double predY = this.Predict(dataX[i]);
        double actualY = dataY[i];
        sum += (predY - actualY) *
          (predY - actualY);
      }
      return sum / n;
    }

    // ------------------------------------------------------------------------

    public void SetWeights(double[] weights)
    {
      // used for a Load function, not-yet-implemented
      int ptr = 0;
      for (int i = 0; i < this.numInput; ++i)
        for (int j = 0; j < this.numHiddenA; ++j)
          this.iaWeights[i][j] = weights[ptr++];

      for (int i = 0; i < numHiddenA; ++i)
        this.aBiases[i] = weights[ptr++];

      for (int i = 0; i < this.numHiddenA; ++i)
        for (int j = 0; j < this.numHiddenB; ++j)
          this.abWeights[i][j] = weights[ptr++];

      for (int i = 0; i < this.numHiddenB; ++i)
        this.bBiases[i] = weights[ptr++];

      for (int i = 0; i < this.numHiddenB; ++i)
        for (int j = 0; j < this.numOutput; ++j)
          this.boWeights[i][j] = weights[ptr++];

      for (int i = 0; i < this.numOutput; ++i)
        this.oBiases[i] = weights[ptr++];
    }

    // ------------------------------------------------------------------------

  } // class NeuralNetworkDeepRegressor

  // ==========================================================================

} // ns