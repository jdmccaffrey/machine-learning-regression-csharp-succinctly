using System;
using System.IO;
using System.Collections.Generic;

namespace KernelRidgeRegressionSGD
{
  internal class KernelRidgeRegressionSGDProgram
  {
    static void Main(string[] args)
    {
      Console.WriteLine("\nBegin Kernel Ridge Regression using C# ");

      // 1. load data
      Console.WriteLine("\nLoading synthetic train (200) and test (40) data ");
      string trainFile = "C:\\VSR\\Data\\synthetic_train_200.txt";
      int[] colsX = new int[] { 0, 1, 2, 3, 4 };
      double[][] trainX = MatLoad(trainFile, colsX, ',', "#");
      double[] trainY = MatToVec(MatLoad(trainFile, [5], ',', "#"));

      string testFile = "C:\\VSR\\Data\\synthetic_test_40.txt";
      double[][] testX = MatLoad(testFile, colsX, ',', "#");
      double[] testY = MatToVec(MatLoad(testFile, [5], ',', "#"));

      Console.WriteLine("\nFirst three train X: ");
      for (int i = 0; i < 3; ++i)
        VecShow(trainX[i], 4, 8);

      Console.WriteLine("\nFirst three train y: ");
      for (int i = 0; i < 3; ++i)
        Console.WriteLine(trainY[i].ToString("F4").PadLeft(8));

      // 2. create model and train using SGD
      Console.WriteLine("\nCreating KRR model ");
      double gamma = 0.30;      // RBF param
      double alpha = 0.000010;  // L2 weight decay not same as regular L2 alpha
      Console.WriteLine("Setting RBF gamma = " + gamma.ToString("F2"));
      Console.WriteLine("Setting L2 decay alpha =  " + alpha.ToString("F5"));
      KernelRidgeRegressor model = 
        new KernelRidgeRegressor(gamma, alpha, seed: 99);
      Console.WriteLine("Done ");

      Console.WriteLine("\nTraining KRR model using SGD ");
      double lrnRate = 0.05;
      int maxEpochs = 2000;
      Console.WriteLine("Setting lrnRate = " + lrnRate.ToString("F4"));
      Console.WriteLine("Setting maxEpochs = " + maxEpochs);

      model.Train(trainX, trainY, lrnRate, maxEpochs);
      Console.WriteLine("Done ");

      // 3. examine model weights
      Console.WriteLine("\nModel weights: ");
      VecShow(model.weights, 4, 9);

      // 4. evaluate model
      Console.WriteLine("\nEvaluating model ");
      double accTrain = model.Accuracy(trainX, trainY, 0.05);
      Console.WriteLine("\nAccuracy train (within 0.05) = " +
        accTrain.ToString("F4"));
      double accTest = model.Accuracy(testX, testY, 0.05);
      Console.WriteLine("Accuracy test (within 0.05) = " +
        accTest.ToString("F4"));

      double mseTrain = model.MSE(trainX, trainY);
      Console.WriteLine("\nMSE train = " + mseTrain.ToString("F4"));
      double mseTest = model.MSE(testX, testY);
      Console.WriteLine("MSE test = " + mseTest.ToString("F4"));

      // 5. use model
      double[] x = trainX[0];
      Console.WriteLine("\nPredicting for x = ");
      VecShow(x, 4, 9);
      double predY = model.Predict(x);
      Console.WriteLine("Predicted y = " + predY.ToString("F4"));

      Console.WriteLine("\nEnd demo ");
      Console.ReadLine();
    } // Main()

    // ----------------------------------------------------------------
    // helpers for Main(): MatLoad, MatToVec, VecShow
    // ----------------------------------------------------------------

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

    public static void VecShow(double[] vec, int dec, int wid)
    {
      for (int i = 0; i < vec.Length; ++i)
        Console.Write(vec[i].ToString("F" + dec).PadLeft(wid));
      Console.WriteLine("");
    }
  } // Program

  // ==========================================================================

  public class KernelRidgeRegressor
  {
    public double gamma;       // for RBF kernel
    public double alpha;       // L2 regularization or weight decay if SGD)
    public double[][] trainX;  // need for any prediction
    public double[] trainY;    // for debugging
    public double[] weights;   // one per trainX item
    private Random rnd;

    // ------------------------------------------------------------------------

    public KernelRidgeRegressor(double gamma, double alpha, int seed = 0)
    {
      this.gamma = gamma;
      this.alpha = alpha;
      this.trainX = new double[0][];  // sort-of null
      this.trainY = new double[0];    // keep compiler happy
      this.weights = new double[0];   // allocated in Train()
      this.rnd = new Random(seed);    // used if SGD training
    }

    // ------------------------------------------------------------------------

    public void Train(double[][] trainX, double[] trainY,
      double lrnRate, int maxEpochs)
    {
      int freq = maxEpochs / 5;  // when to show progress

      // 0. store trainX -- needed by Predict()
      this.trainX = trainX;  // by ref -- could make explicit copy 
      this.trainY = trainY;  // not used this version

      this.weights = new double[trainX.Length];
      double lo = -0.01; double hi = 0.01;
      for (int i = 0; i < this.weights.Length; ++i)
        this.weights[i] = (hi - lo) * this.rnd.NextDouble() + lo;

      // 1. set up indices for shuffling
      int[] indices = new int[trainX.Length];
      for (int i = 0; i < indices.Length; ++i)
        indices[i] = i;

      for (int epoch = 0; epoch < maxEpochs; ++epoch)
      {
        // inline shuffle
        for (int i = 0; i < indices.Length; ++i)
        {
          int ri = this.rnd.Next(i, indices.Length);
          int tmp = indices[i];
          indices[i] = indices[ri];
          indices[ri] = tmp;
        }

        for (int i = 0; i < trainX.Length; ++i)
        {
          int idx = indices[i];
          double[] x = trainX[idx];
          double predY = this.Predict(x);
          double actualY = trainY[idx];
          // weight decay before update
          this.weights[idx] *= (1.0 - this.alpha);
          // update weight associated with x
          this.weights[idx] -= lrnRate * (predY - actualY);
        } // each item

        if (epoch % freq == 0)
        {
          double mse = this.MSE(trainX, trainY);
          string s1 = "epoch = " + epoch.ToString().PadLeft(6);
          string s2 = "   MSE = " + mse.ToString("F4");
          Console.WriteLine(s1 + s2);
        }

      } // each epoch
    }

    // ------------------------------------------------------------------------

    public double Predict(double[] x)
    {
      int n = this.trainX.Length;
      double sum = 0.0;
      for (int i = 0; i < n; ++i)
      {
        double[] currX = this.trainX[i];
        double k = Rbf(x, currX, this.gamma);
        sum += this.weights[i] * k;
      }
      return sum; // no explicit bias
    }

    // ------------------------------------------------------------------------

    private static double Rbf(double[] v1, double[] v2, double gamma)
    {
      // the gamma version as opposed to sigma version
      int dim = v1.Length;
      double sum = 0.0;
      for (int i = 0; i < dim; ++i) // squared Euclidean distance
      {
        sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
      }
      return Math.Exp(-1 * gamma * sum); 
    }

    // ------------------------------------------------------------------------

    public double Accuracy(double[][] dataX, double[] dataY, double pctClose)
    {
      int numCorrect = 0; int numWrong = 0;
      int n = dataX.Length;

      for (int i = 0; i < n; ++i)
      {
        double[] x = dataX[i];
        double predY = this.Predict(x);
        double actualY = dataY[i];
        if (Math.Abs(predY - actualY) < Math.Abs(pctClose * actualY))
          numCorrect += 1;
        else
          numWrong += 1;
      }
      return (numCorrect * 1.0) / (numCorrect + numWrong);
    }

    // ------------------------------------------------------------------------

    public double MSE(double[][] dataX, double[] dataY)
    {
      double sum = 0.0;
      int n = dataX.Length;
      for (int i = 0; i < n; ++i)
      {
        double[] x = dataX[i];
        double actualY = dataY[i];
        double predY = this.Predict(x);
        sum += (actualY - predY) * (actualY - predY);
      }
      return sum / n;
    }

  } // class

  // ==========================================================================

} // ns