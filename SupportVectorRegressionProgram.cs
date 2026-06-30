using System;
using System.IO;
using System.Collections.Generic;

// kernelized SVR with SGD training
// hard-wired RBF kernel function

namespace SupportVectorRegression
{
  internal class SupportVectorRegressionProgram
  {
    static void Main(string[] args)
    {
      Console.WriteLine("\nBegin support vector regression with SSGD ");

      // 1. load data
      Console.WriteLine("\nLoading train (200) and test (40) data ");
      string trainFile = "C:\\VSR\\Data\\synthetic_train_200.txt";
      double[][] trainX = Utils.MatLoad(trainFile, 
        new int[] { 0, 1, 2, 3, 4 }, ',', "#");
      double[] trainY = Utils.MatToVec(Utils.MatLoad(trainFile,
        new int[] { 5 }, ',', "#"));

      string testFile = "C:\\VSR\\Data\\synthetic_test_40.txt";
      double[][] testX = Utils.MatLoad(testFile,
        new int[] { 0, 1, 2, 3, 4 }, ',', "#");
      double[] testY = Utils.MatToVec(Utils.MatLoad(testFile,
        new int[] { 5 }, ',', "#"));
      Console.WriteLine("Done ");

      Console.WriteLine("\nFirst three X predictors: ");
      for (int i = 0; i < 3; ++i)
        Utils.VecShow(trainX[i], 4, 9);
      Console.WriteLine("\nFirst three target y: ");
      for (int i = 0; i < 3; ++i)
        Console.WriteLine(trainY[i].ToString("F4").PadLeft(8));

      // 2. create model
      Console.WriteLine("\nCreating SVR object");
      double gamma = 0.30;    // RBF param
      double epsilon = 0.010;
      double C = 1.0;
      double lrnRate = 0.001;
      int maxEpochs = 5000;
      double tol = 1.0e-4;  // defines 0-weight

      Console.WriteLine("Setting RBF gamma = " + gamma.ToString("F4"));
      Console.WriteLine("Setting epsilon = " + epsilon.ToString("F6"));
      Console.WriteLine("Setting C = " + C.ToString("F2"));
      Console.WriteLine("Setting lrnRate = " + lrnRate.ToString("F4"));
      Console.WriteLine("Setting maxEpochs = " + maxEpochs);
      Console.WriteLine("Setting tol = " + tol.ToString("F6"));

      SupportVectorRegressor model = 
        new SupportVectorRegressor(gamma, epsilon, C, lrnRate, maxEpochs,
        tol, seed: 0);
      Console.WriteLine("Done ");

      // 3. train model
      Console.WriteLine("\nTraining SVR model using SSGD ");
      model.Train(trainX, trainY);
      Console.WriteLine("Done ");

      // 4. inspect trained model
      Console.WriteLine("\nModel alpha (weights): ");
      Utils.VecShow(model.alpha, 4, 9);
      Console.WriteLine("\nModel bias = " + model.b.ToString("F4"));
      Console.WriteLine("\nNumber support vectors = " + model.suppX.Length);

      // 5. evaluate model
      Console.WriteLine("\nEvaluating model ");
      double trainAcc = model.Accuracy(trainX, trainY, 0.05);
      double testAcc = model.Accuracy(testX, testY, 0.05);

      Console.WriteLine("\nTrain acc (within 0.05) = " + 
        trainAcc.ToString("F4"));
      Console.WriteLine("Test acc (within 0.05) = " +
        testAcc.ToString("F4"));

      double trainMSE = model.MSE(trainX, trainY);
      double testMSE = model.MSE(testX, testY);

      Console.WriteLine("\nTrain MSE = " + trainMSE.ToString("F4"));
      Console.WriteLine("Test MSE = " + testMSE.ToString("F4"));

      // 6. use model
      double[] x = trainX[0];
      Console.WriteLine("\nPredicting for x = ");
      Utils.VecShow(x, 4, 9);
      double predY = model.Predict(x);
      Console.WriteLine("Predicted y = " + predY.ToString("F4"));

      Console.WriteLine("\nEnd SVR with SSGD training demo ");
      Console.ReadLine();
    } // Main()

  } // class Program

  // ==========================================================================

  public class SupportVectorRegressor
  {
    public double gamma;      // for RBF kernel
    public double epsilon;
    public double C;          // complexity regularization
    public double[][] suppX;  // support vectors
    public double[] suppY;
    public double[] alpha;    // one per trainX item
    public double b;          // bias
    public double lrnRate;    // for SGD training
    public int maxEpochs;
    public double tol;        // defines a zero alpha-weight
    public Random rnd;

    // ------------------------------------------------------------------------

    public SupportVectorRegressor(double gamma, double epsilon, double C,
      double lrnRate, int maxEpochs, double tol, int seed = 0)
    {
      this.gamma = gamma;
      this.epsilon = epsilon;
      this.C = C;
      this.suppX = new double[0][]; // compiler happy
      this.suppY = new double[0];
      this.lrnRate = lrnRate;
      this.maxEpochs = maxEpochs;
      this.tol = tol;
      this.alpha = new double[0];
      this.b = 0.0;
      this.rnd = new Random(seed);  // shuffle train order
    } // ctor

    // ------------------------------------------------------------------------

    public void Train(double[][] trainX, double[] trainY)
    {
      this.suppX = trainX;
      this.suppY = trainY;
      int n = trainX.Length;

      // init weights
      this.alpha = new double[n];
      double lo = -0.01; double hi = 0.01; 
      for (int i = 0; i < n; ++i)
        this.alpha[i] = (hi - lo) * this.rnd.NextDouble() + lo;
      this.b = 0.0;

      // precompute all rbf values to K for fast train
      // not feasible for huge datasets
      double[][] K = this.MakeK(trainX);

      // set up indices for random order SGD training
      int[] indices = Utils.VecRange(n); // 0, 1, 2, ..
      double lamda = 1.0 / this.C;
      int progressFreq = (int)(this.maxEpochs / 5);

      // main sub-gradient processing loop
      for (int epoch = 0; epoch < this.maxEpochs; ++epoch)
      {
        this.Shuffle(indices);
        for (int i = 0; i < indices.Length; ++i)
        {
          int idx = indices[i];
          double predY = 0.0;
          for (int j = 0; j < this.alpha.Length; ++j)
            predY += this.alpha[j] * K[idx][j]; // fast
          predY += this.b;
          double error = predY - trainY[idx];

          double gradLoss;
          bool insideTube = false;
          if (error > this.epsilon)
            gradLoss = 1.0;
          else if (error < -this.epsilon)
            gradLoss = -1.0;
          else
          {
            gradLoss = 0.0;
            insideTube = true;
          }

          // local kernel regularization gradient
          double gradReg = this.alpha[idx] * K[idx][idx];

          //  decoupled updates to the active index
          this.alpha[idx] -= this.lrnRate *
            (lamda * gradReg + gradLoss);
          this.b -= this.lrnRate * gradLoss;

          // force tiny weights to 0
          if (insideTube == true &&
            Math.Abs(this.alpha[idx]) < this.tol)
            this.alpha[idx] = 0.0;

          // in-loop clip to bound updates mid-flight
          if (this.alpha[idx] < -this.C)
            this.alpha[idx] = -this.C;
          else if (this.alpha[idx] > this.C)
            this.alpha[idx] = this.C;

        } // each item

        // show training progress every few epochs
        if (epoch % progressFreq == 0)
        {
          double mse = this.MSE(trainX, trainY);
          double acc = this.Accuracy(trainX, trainY, 0.05);
          string s1 = "epoch = " + epoch.ToString().PadLeft(6);
          string s2 = " MSE = " + mse.ToString("F4");
          string s3 = " acc = " + acc.ToString("F4");
          Console.WriteLine(s1 + s2 + s3);
        }

      } // each epoch

      // final global clip
      for (int i = 0; i < n; ++i)
      {
        if (this.alpha[i] < -this.C)
          this.alpha[i] = -this.C;
        else if (this.alpha[i] > this.C)
          this.alpha[i] = this.C;
      }

      // prune: store only explicit support vectors
      List<int> svLst = new List<int>();
      for (int i = 0; i < this.alpha.Length; ++i)
      {
        if (Math.Abs(this.alpha[i]) > 1.0e-5)
          svLst.Add(i);
      }
      int[] svMask = svLst.ToArray();

      this.suppX = Utils.MatSelectRows(trainX, svMask);
      this.suppY = Utils.VecSelectItems(trainY, svMask);
      this.alpha = Utils.VecSelectItems(this.alpha, svMask);

      return;  // all done
    } // Train

    // ------------------------------------------------------------------------

    public double Predict(double[] x)
    {
      int n = this.suppX.Length;
      double sum = 0.0;
      for (int i = 0; i < n; ++i)
      {
        double[] sv = this.suppX[i];
        double k = this.RBF(x, sv);
        sum += this.alpha[i] * k;
      }
      return sum + this.b;
    }

    // ------------------------------------------------------------------------

    public double Accuracy(double[][] dataX,
      double[] dataY, double pctClose)
    {
      int numCorrect = 0; int numWrong = 0;
      int n = dataX.Length;
      for (int i = 0; i < n; ++i)
      {
        double[] x = dataX[i];
        double actualY = dataY[i];
        double predY = this.Predict(x);
        if (Math.Abs(actualY - predY) < Math.Abs(actualY * pctClose))
          ++numCorrect;
        else
          ++numWrong;
      }
      return (numCorrect * 1.0) / n;
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

    // ----------------------------------------------------------------

    private void Shuffle(int[] indices)
    {
      // Fisher-Yates helper for Train()
      for (int i = 0; i < indices.Length; ++i)
      {
        int ri = this.rnd.Next(i, indices.Length);
        int tmp = indices[i];
        indices[i] = indices[ri];
        indices[ri] = tmp;
      }
    } // Shuffle

    // ----------------------------------------------------------------

    private double RBF(double[] v1, double[] v2)
    {
      int n = v1.Length;
      double sum = 0.0;
      for (int i = 0; i < n; ++i)
      {
        double d = v1[i] - v2[i];
        sum += d * d;
      }
      double result = Math.Exp(-1 * this.gamma * sum);
      return result;
    }

    // ----------------------------------------------------------------

    private double[][] MakeK(double[][] X)
    {
      // Kernel-Gram matrix helper for Train()
      // pre-compute all similarities, to avoid re-computes
      int n = X.Length;
      double[][] result = Utils.MatMake(n, n);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          result[i][j] = this.RBF(X[i], X[j]);
      return result;
    }

  } // class

  // ==========================================================================

  public class Utils
  {
    // ------------------------------------------------------------------------

    public static double[][] MatLoad(string fn,
      int[] usecols, char sep, string comment)
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

    // ------------------------------------------------------------------------

    public static double[] MatToVec(double[][] X)
    {
      int nRows = X.Length;
      int nCols = X[0].Length;
      double[] result = new double[nRows * nCols];
      int k = 0;
      for (int i = 0; i < nRows; ++i)
        for (int j = 0; j < nCols; ++j)
          result[k++] = X[i][j];
      return result;
    }

    // ------------------------------------------------------------------------

    public static double[][] MatSelectRows(double[][] X,
      int[] rows)
    {
      int nRowsSrc = X.Length;
      int nColsSrc = X[0].Length;
      int n = rows.Length;
      double[][] result = MatMake(n, nColsSrc);

      for (int i = 0; i < n; ++i) // i pts into result
      {
        int srcRow = rows[i];
        for (int j = 0; j < nColsSrc; ++j)
          result[i][j] = X[srcRow][j];
      }
      return result;
    }

    // ------------------------------------------------------------------------

    public static double[] VecSelectItems(double[] vec,
      int[] idxs)
    {
      int n = idxs.Length;
      double[] result = new double[n];
      for (int i = 0; i < n; ++i)
        result[i] = vec[idxs[i]];
      return result;
    }

    // ------------------------------------------------------------------------

    public static double[][] MatMake(int nRows, int nCols)
    {
      double[][] result = new double[nRows][];
      for (int i = 0; i < nRows; ++i)
        result[i] = new double[nCols];
      return result;
    }

    // ------------------------------------------------------------------------

    public static double VecMean(double[] vec)
    {
      int n = vec.Length;
      double sum = 0.0;
      for (int i = 0; i < n; ++i)
        sum += vec[i];
      double result = sum / n;
      return result;
    }

    // ------------------------------------------------------------------------

    public static int[] VecRange(int n)
    {
      int[] result = new int[n];
      for (int i = 0; i < n; ++i)
        result[i] = i;
      return result;
    }

    // ------------------------------------------------------------------------

    public static void VecShow(int[] vec, int wid)
    {
      for (int i = 0; i < vec.Length; ++i)
        Console.Write(vec[i].ToString().PadLeft(wid));
      Console.WriteLine("");
    }

    // ------------------------------------------------------------------------
    
    public static void VecShow(double[] vec, int dec, int wid)
    {
      for (int i = 0; i < vec.Length; ++i)
        Console.Write(vec[i].ToString("F" + dec).PadLeft(wid));
      Console.WriteLine("");
    }

    // ------------------------------------------------------------------------

    public static void MatShow(double[][] M, int dec, int wid)
    {
      int nRows = M.Length; int nCols = M[0].Length;
      double small = 1.0 / Math.Pow(10, dec);
      for (int i = 0; i < nRows; ++i)
      {
        for (int j = 0; j < nCols; ++j)
        {
          double v = M[i][j];
          if (Math.Abs(v) < small) v = 0.0;
          Console.Write(v.ToString("F" + dec).
            PadLeft(wid));
        }
        Console.WriteLine("");
      }
    }

  } // class Utils

  // ============================================================================

} // ns
