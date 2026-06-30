using System;
using System.IO;
using System.Collections.Generic;

// simplified, one hidden layer, "online" train
// hard-coded tanh hidden activation, identity output

namespace NeuralNetworkRegression
{
  internal class NeuralNetworkRegressionProgram
  {
    static void Main(string[] args)
    {
      Console.WriteLine("\nBegin NN regression using C# ");

      // 1. load data
      Console.WriteLine("\nLoading synthetic train (200) and test (40) data ");
      string trainFile = "C:\\VSR\\Data\\synthetic_train_200.txt";
      double[][] trainX = MatLoad(trainFile, new int[] { 0, 1, 2, 3, 4 },
        ',', "#");
      double[] trainY = MatToVec(MatLoad(trainFile, new int[] { 5 },
        ',', "#"));

      string testFile = "C:\\VSR\\Data\\synthetic_test_40.txt";
      double[][] testX = MatLoad(testFile, new int[] { 0, 1, 2, 3, 4 },
        ',', "#");
      double[] testY = MatToVec(MatLoad(testFile,  new int[] { 5 },
        ',', "#"));
      Console.WriteLine("Done ");

      Console.WriteLine("\nFirst three train X: ");
      for (int i = 0; i < 3; ++i)
        VecShow(trainX[i], 4, 8);

      Console.WriteLine("\nFirst three train y: ");
      for (int i = 0; i < 3; ++i)
        Console.WriteLine(trainY[i].ToString("F4").PadLeft(8));

      // 2. create NN
      Console.WriteLine("\nCreating 5-12-1 tanh() identity() neural network ");
      NeuralNetworkRegressor nn = new NeuralNetworkRegressor(5, 12, 1);
      Console.WriteLine("Done ");

      // 3. train NN
      double lrnRate = 0.11;
      int maxEpochs = 10000;
      double decay = 0.0000001;
     
      Console.WriteLine("\nSetting lrnRate = " + lrnRate.ToString("F4"));
      Console.WriteLine("Setting maxEpochs = " + maxEpochs);
      Console.WriteLine("Setting decay = " + decay.ToString("0.0e+0"));

      Console.WriteLine("\nStarting training ");
      nn.Train(trainX, trainY, lrnRate, maxEpochs, decay);
      Console.WriteLine("Done ");

      // 4. examine model weights and biases
      Console.WriteLine("\ninput-to-hidden weights: ");
      for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 12; ++j)
          Console.Write(nn.ihWeights[i][j].ToString("F4").PadLeft(9));
      Console.WriteLine("");

      Console.WriteLine("\nhidden node biases: ");
      for (int j = 0; j < 12; ++j)
        Console.Write(nn.hBiases[j].ToString("F4").PadLeft(9));
      Console.WriteLine("");

      Console.WriteLine("\nhidden-to-output weights: ");
      for (int j = 0; j < 12; ++j)
        for (int k = 0; k < 1; ++k)
          Console.Write(nn.hoWeights[j][k].ToString("F4").PadLeft(9));
      Console.WriteLine("");

      Console.WriteLine("\noutput node bias(es): ");
      for (int k = 0; k < 1; ++k)
        Console.Write(nn.oBiases[k].ToString("F4").PadLeft(9));
      Console.WriteLine("");

      // 5. evaluate trained model
      Console.WriteLine("\nEvaluating model ");
      double trainAcc = nn.Accuracy(trainX, trainY, 0.05);
      Console.WriteLine("\nAccuracy (5%) on train data = " +
        trainAcc.ToString("F4"));

      double testAcc = nn.Accuracy(testX, testY, 0.05);
      Console.WriteLine("Accuracy (5%) on test data  = " +
        testAcc.ToString("F4"));

      double trainMSE = nn.MSE(trainX, trainY);
      Console.WriteLine("\nMSE on train data = " +
        trainMSE.ToString("F4"));

      double testMSE = nn.MSE(testX, testY);
      Console.WriteLine("MSE on test data = " +
        testMSE.ToString("F4"));

      // 6. use model
      double[] x = trainX[0];
      Console.WriteLine("\nPredicting for x = ");
      VecShow(x, 4, 9);
      double predY = nn.Predict(x);
      Console.WriteLine("Predicted y = " + predY.ToString("F4"));

      Console.WriteLine("\nEnd demo ");
      Console.ReadLine();
    } // Main()

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

  } // class Program

  // ==========================================================================

  public class NeuralNetworkRegressor
  {
    public int ni; // number input nodes
    public int nh; // hidden
    public int no; // outout

    public Random rnd; // wt init, train order

    public double[] iNodes;
    public double[] hNodes;
    public double[] oNodes;  // single val as array

    public double[][] ihWeights; // input-hidden
    public double[][] hoWeights; // hidden-output

    public double[] hBiases; // hidden node biases
    public double[] oBiases; // output node bias(es)

    // ------------------------------------------------------------------------

    public NeuralNetworkRegressor(int numIn, int numHid, int numOut,
      int seed = 0)
    {
      this.ni = numIn;
      this.nh = numHid;
      this.no = numOut;  // 1 for regression

      this.iNodes = new double[numIn];
      this.hNodes = new double[numHid];
      this.oNodes = new double[numOut];  // [1]

      this.ihWeights = MatMake(numIn, numHid);
      this.hoWeights = MatMake(numHid, numOut);

      this.hBiases = new double[numHid];
      this.oBiases = new double[numOut];  // [1]

      this.rnd = new Random(seed);
    } // ctor

    // ------------------------------------------------------------------------

    public double Predict(double[] x)
    {
      for (int i = 0; i < x.Length; ++i)
        this.iNodes[i] = x[i];

      // compute hidden nodes
      for (int j = 0; j < this.nh; ++j)
      {
        double sum = 0.0;
        for (int i = 0; i < this.ni; ++i)
          sum += this.iNodes[i] * this.ihWeights[i][j];
        sum += this.hBiases[j];
        this.hNodes[j] = HyperTan(sum);
      }

      // compute output nodes
      for (int k = 0; k < this.no; ++k)
      {
        double sum = 0.0;
        for (int j = 0; j < this.nh; ++j)
          sum += this.hNodes[j] * this.hoWeights[j][k];
        sum += this.oBiases[k];
        this.oNodes[k] = Identity(sum);
      }

      return this.oNodes[0];  // single value
    }

    // ------------------------------------------------------------------------
    
    public void Train(double[][] trainX, double[] trainY,
      double lrnRate, int maxEpochs, double decay)
    {
      int freq = maxEpochs / 5;  // show progress 5 times

      // set up gradients
      double[][] ihGrads = MatMake(this.ni, this.nh);
      double[] hbGrads = new double[this.nh];
      double[][] hoGrads = MatMake(this.nh, this.no);
      double[] obGrads = new double[this.no];

      // set up signals, just for computing convenience
      double[] oSignals = new double[this.no];
      double[] hSignals = new double[this.nh];

      // 1. initialize the weights (no need to init biases)
      double lo = -0.01; double hi = +0.01;

      for (int i = 0; i < this.ni; ++i) // ih weights
        for (int j = 0; j < this.nh; ++j)
          this.ihWeights[i][j] =
            (hi - lo) * this.rnd.NextDouble() + lo;

      for (int i = 0; i < this.nh; ++i) // ho weights
        for (int j = 0; j < this.no; ++j)
          this.hoWeights[i][j] =
            (hi - lo) * this.rnd.NextDouble() + lo;

      // 2. prepare indices for random order processing
      int n = trainX.Length;
      int[] indices = new int[n];
      for (int i = 0; i < n; ++i)
        indices[i] = i;

      // 3. loop max epochs times
      for (int epoch = 0; epoch < maxEpochs; ++epoch)
      {
        this.Shuffle(indices);
        for (int ii = 0; ii < n; ++ii) // loop each item
        {
          int idx = indices[ii];
          double[] x = trainX[idx];
          double actualY = trainY[idx];
          double predY = this.Predict(x); // forward pass

          // 1. compute output node signals
          for (int k = 0; k < this.no; ++k) // k is always 1
          {
            double derivative = 1.0; // for Identity output
            oSignals[k] = derivative * (predY - actualY);
          }

          // 2. hidden-to-output weight gradients
          for (int j = 0; j < this.nh; ++j)
            for (int k = 0; k < this.no; ++k)
              hoGrads[j][k] = oSignals[k] * this.hNodes[j];

          // 3. output node bias gradient
          for (int k = 0; k < this.no; ++k)
            obGrads[k] = oSignals[k] * 1.0; // dummy input

          // 4. hidden node signals
          for (int j = 0; j < this.nh; ++j)
          {
            double sum = 0.0;
            for (int k = 0; k < this.no; ++k)
              sum += oSignals[k] * this.hoWeights[j][k];
            // derivative of tanh(x) = (1 - x) * (1 + x)
            double derivative = (1 - this.hNodes[j]) * (1 + this.hNodes[j]);
            hSignals[j] = derivative * sum;
          }

          // 5. compute input-to-hidden weight gradients
          for (int i = 0; i < this.ni; ++i)
            for (int j = 0; j < this.nh; ++j)
              ihGrads[i][j] = hSignals[j] * this.iNodes[i];

          // 6. hidden node bias gradients
          for (int j = 0; j < this.nh; ++j)
            hbGrads[j] = hSignals[j] * 1.0; // dummy input

          // decay weights slightly before updating

          // 1. decay input-to-hidden weights
          for (int i = 0; i < this.ni; ++i)
            for (int j = 0; j < this.nh; ++j)
              this.ihWeights[i][j] *= (1 - decay);
                    
          // 3. hidden-to-output weights
          for (int j = 0; j < this.nh; ++j)
            for (int k = 0; k < this.no; ++k)
              this.hoWeights[j][k] *= (1 - decay);
       
          // now update weights, biases using gradients

          // 1. update input-to-hidden weights
          for (int i = 0; i < this.ni; ++i)
            for (int j = 0; j < this.nh; ++j)
              this.ihWeights[i][j] -= lrnRate * ihGrads[i][j];

          // 2. hidden node biases
          for (int j = 0; j < this.nh; ++j)
            this.hBiases[j] -= lrnRate * hbGrads[j];
          
          // 3. hidden-to-output weights
          for (int j = 0; j < this.nh; ++j)
            for (int k = 0; k < this.no; ++k)
              this.hoWeights[j][k] -= lrnRate * hoGrads[j][k];
          
          // 4. output node biases
          for (int k = 0; k < this.no; ++k)
            this.oBiases[k] -= lrnRate * obGrads[k];
          
        } // ii each item

        if (epoch % freq == 0)  // show progress
        {
          double mse = this.MSE(trainX, trainY);
          double acc = this.Accuracy(trainX, trainY, 0.05);

          string s1 = "epoch: " + epoch.ToString().PadLeft(6);
          string s2 = "  MSE = " + mse.ToString("F4");
          string s3 = "  acc = " + acc.ToString("F4");
          Console.WriteLine(s1 + s2 + s3);
        }

      } // each epoch

      return; // weights and biases have been set
    } // Train()

    // ------------------------------------------------------------------------

    public double Accuracy(double[][] dataX,
      double[] dataY, double pctClose)
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

    private static double[][] MatMake(int nRows, int nCols)
    {
      double[][] result = new double[nRows][];
      for (int i = 0; i < nRows; ++i)
        result[i] = new double[nCols];
      return result;
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

    private static double HyperTan(double x) // avoid extreme values
    {
      if (x < -8.0) return -1.0;
      else if (x > 8.0) return 1.0;
      else return Math.Tanh(x);
    }

    // ------------------------------------------------------------------------

    private static double Identity(double x)
    {
      return x;
    }

    // ------------------------------------------------------------------------

  } // class NeuralNetworkRegressor

} // ns
