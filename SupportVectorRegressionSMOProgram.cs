using System;
using System.IO;
using System.Collections.Generic;

// SVR using sequential minimal optimization training (SMO)
// (libsvm library is very complicated, standard)

namespace SupportVectorRegressionSMO
{
  internal class SupportVectorRegressionSMOProgram
  {
    static void Main(string[] args)
    {
      Console.WriteLine("\nBegin support vector regression with SMO ");

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

      // 2. create SVR model
      Console.WriteLine("\nCreating SVR object");
      double gamma = 0.30;    // RBF param controls similarity function
      double epsilon = 0.010; // defines supp vecs and non-supp vecs
      double C = 1.0;         // complexity regularization
      double tol = 0.0290;    // KKT tolerance is ultra sensitive
      int maxPasses = 10;     // number times no change

      Console.WriteLine("Setting RBF gamma = " + gamma.ToString("F2"));
      Console.WriteLine("Setting epsilon = " + epsilon.ToString("F4"));
      Console.WriteLine("Setting C = " + C.ToString("F2"));
      Console.WriteLine("Setting KKT tol = " + tol.ToString("F6"));
      Console.WriteLine("Setting maxPasses = " + maxPasses);

      SupportVectorRegressor model =
        new SupportVectorRegressor(gamma, epsilon, C, tol, maxPasses);
      Console.WriteLine("Done ");

      // 3. train model
      Console.WriteLine("\nTraining SVR model using SMO ");
      model.Train(trainX, trainY);
      Console.WriteLine("Done ");

      // 4. inspect model
      Console.WriteLine("\nModel dual weights: ");
      Utils.VecShow(model.dualWeights, 4, 9);
      Console.WriteLine("\nModel bias = " + model.b.ToString("F4"));
      Console.WriteLine("\nNumber supp vectors = " + model.dualWeights.Length);

      // 5. evaluate trained model
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
      Console.WriteLine("\nPredicting for trainX[0] ");
      double y = model.Predict(x);
      Console.WriteLine("Predicted y = " + y.ToString("F4"));

      Console.WriteLine("\nEnd SVR with SMO training demo ");
      Console.ReadLine();
    } // Main()

  } // class

  // ==========================================================================

  public class SupportVectorRegressor
  {
    public double gamma;  // for RBF kernel
    public double epsilon;
    public double C; // regularization
    public double tol;  // for KKT conditions
    public int maxPasses;  // times no improvement
    public double[][] suppX;
    public double[] suppY;
    public double[] alpha;
    public double[] alphaStar;
    public double[] dualWeights; // alpha* - alpha
    public double b; // bias
    public Random rnd;  // training is probabilistic

    // ------------------------------------------------------------------------

    public SupportVectorRegressor(double gamma, double epsilon, double C,
      double tol, int maxPasses, int seed = 0)
    {
      this.gamma = gamma;
      this.epsilon = epsilon;
      this.C = C;
      this.tol = tol;
      this.maxPasses = maxPasses;

      this.suppX = new double[0][]; // compiler happy
      this.suppY = new double[0];
      this.alpha = new double[0];
      this.alphaStar = new double[0];
      this.dualWeights = new double[0];
      this.b = 0.0;
      this.rnd = new Random(seed); // SMO selection
    }

    // ------------------------------------------------------------------------

    public void Train(double[][] trainX, double[] trainY)
    {
      // train using SMO algorithm
      this.suppX = trainX;  // needed by Predict()
      this.suppY = trainY;  // not used this version
      int n = trainX.Length;

      this.alpha = new double[n];
      this.alphaStar = new double[n];
      this.b = Utils.VecMean(trainY);

      // precompute all rbf values to K for fast train
      double[][] K = this.MakeK(trainX);

      int nPasses = 0;  // number times no change in alpha
      while (nPasses < this.maxPasses)
      {
        int numChangedAlphas = 0;

        for (int i = 0; i < n; ++i) // walk each train item
        {
          double[] iCurrCol = Utils.MatGetColumn(K, i);
          double iPred =
            Utils.VecDot(Utils.VecSubtract(this.alphaStar,
            this.alpha), iCurrCol) + this.b;
          double iErr = iPred - trainY[i];

          // check KKT conditions to see if [i] warrants update
          if ((iErr > this.epsilon + this.tol && this.alpha[i] < this.C) ||
            (iErr > this.epsilon + this.tol && this.alphaStar[i] > 0.0) ||
            (iErr < -this.epsilon - this.tol && this.alpha[i] > 0.0) ||
            (iErr < -this.epsilon - this.tol && this.alphaStar[i] < this.C))
          {
            // update to [i] is warranted
            // pick random j, not the same as i
            int j = i;
            while (j == i)
              j = this.rnd.Next(0, n);

            double[] jCurrCol = Utils.MatGetColumn(K, j);
            double jPred = Utils.VecDot(Utils.VecSubtract(this.alphaStar,
              this.alpha), jCurrCol) + this.b;
            double jErr = jPred - trainY[j];

            // save old alpha and alphaStar
            double iOldAlpha = this.alpha[i];
            double iOldAlphaStar = this.alphaStar[i];
            double jOldAlpha = this.alpha[j];
            double jOldAlphaStar = this.alphaStar[j];

            // kernel second step denominator
            double eta = K[i][i] + K[j][j] - (2.0 * K[i][j]);
            if (eta <= 0.0)
              continue;

            // SVR linear constraint constant
            double constraint = (iOldAlphaStar - iOldAlpha) +
              (jOldAlphaStar - jOldAlpha);

            // joint proxy variable
            double jOldDelta = jOldAlphaStar - jOldAlpha;
            double jNewDelta = jOldDelta + (iErr - jErr) / eta;

            // compute Complexity bounds
            double L = Math.Max(-this.C, constraint - this.C);
            double H = Math.Min(this.C, constraint + this.C);
            // clip
            if (jNewDelta < L)
              jNewDelta = L;
            else if (jNewDelta > H)
              jNewDelta = H;

            if (Math.Abs(jNewDelta - jOldDelta) < 1.0e-5) // could parameterize
              continue;

            // reconstruct j
            if (jNewDelta >= 0.0)
            {
              this.alphaStar[j] = jNewDelta;
              this.alpha[j] = 0.0;
            }
            else
            {
              this.alphaStar[j] = 0.0;
              this.alpha[j] = -jNewDelta;
            }

            // update i
            double iNewValue = constraint - jNewDelta;  // iNewValue
            if (iNewValue >= 0.0)
            {
              this.alphaStar[i] = iNewValue;
              this.alpha[i] = 0.0;
            }
            else
            {
              this.alphaStar[i] = 0.0;
              this.alpha[i] = -iNewValue;
            }

            // update bias
            double b1 = this.b - iErr - ((this.alphaStar[i] -
              this.alpha[i]) - (iOldAlphaStar - iOldAlpha)) *
              K[i][i] - ((this.alphaStar[j] -
              this.alpha[j]) - (jOldAlphaStar - jOldAlpha)) *
              K[i][j];

            double b2 = this.b - jErr - ((this.alphaStar[i] -
              this.alpha[i]) - (iOldAlphaStar - iOldAlpha)) *
              K[i][j] - ((this.alphaStar[j] -
              this.alpha[j]) - (jOldAlphaStar - jOldAlpha)) *
              K[i][j];

            if (0.0 < this.alpha[i] && this.alpha[i] < this.C ||
              0.0 < this.alphaStar[i] && this.alphaStar[i] < this.C)
              this.b = b1;
            else if (0.0 < this.alpha[j] && this.alpha[j] < this.C ||
              0.0 < this.alphaStar[j] && this.alphaStar[j] < this.C)
              this.b = b2;
            else
              this.b = (b1 + b2) / 2.0;

            ++numChangedAlphas;
          }
          else
          {
            ; // no change to alpha[i] warranted
          }

        } // each item i

        if (numChangedAlphas == 0)
          ++nPasses;
        else
          nPasses = 0;  // reset
      } // while

      // prune
      // 1. combine alpha and alpha*
      this.dualWeights = Utils.VecSubtract(this.alphaStar, this.alpha);

      // 2. compute mask based on small dual wts
      List<int> lstMask = new List<int>();
      double maskTol = 1.0e-4;  // larger tol == fewer svs
      for (int i = 0; i < n; ++i)
        if (Math.Abs(this.dualWeights[i]) > maskTol)
          lstMask.Add(i);
      int[] svMask = lstMask.ToArray();

      // 3. mask supp vecs
      this.suppX = Utils.MatSelectRows(trainX, svMask);
      this.suppY = Utils.VecSelectItems(trainY, svMask);

      // 4. mask dual weights
      this.dualWeights = Utils.VecSelectItems(this.dualWeights, svMask);

      // leave source alphas alone . . 

      return;  // all done
    } // Train

    // ------------------------------------------------------------------------

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

    // ------------------------------------------------------------------------

    private double[][] MakeK(double[][] X)
    {
      // Kernel-Gram matrix helper for Train()
      // pre-compute all similarities, to avoid re-computes
      int n = X.Length;
      double[][] result = Utils.MatMake(n, n);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          result[i][j] = this.RBF(this.suppX[i], this.suppX[j]);
      return result;
    }

    // ------------------------------------------------------------------------

    public double Predict(double[] x)
    {
      int n = this.suppX.Length;
      double sum = 0.0;
      for (int i = 0; i < n; ++i)
      {
        double k = this.RBF(this.suppX[i], x);
        sum += this.dualWeights[i] * k;
      }
      return sum + this.b;
    }

    // ------------------------------------------------------------------------

    public double Accuracy(double[][] dataX, double[] dataY, double pctClose)
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
  } // class SupportVectorRegressor

  // ==========================================================================

  public class Utils
  {
    // ------------------------------------------------------------------------

    public static double[][] MatLoad(string fn, int[] usecols,
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

    public static double[] MatGetColumn(double[][] X, int col)
    {
      int nRows = X.Length; int nCols = X[0].Length;
      double[] result = new double[nRows];
      for (int i = 0; i < nRows; ++i)
        result[i] = X[i][col];
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

    public static double[][] MatSelectRows(double[][] X, int[] rows)
    {
      int nRowsSrc = X.Length;
      int nColsSrc = X[0].Length;
      int n = rows.Length;
      double[][] result = MatMake(n, nColsSrc);

      for (int i = 0; i < n; ++i) // i pts into result
      {
        int srcRow = rows[i];
        for (int j = 0; j < nColsSrc; ++j)
        {
          result[i][j] = X[srcRow][j];
        }
      }
      return result;
    }

    // ------------------------------------------------------------------------

    public static double VecMean(double[] vec)
    {
      int n = vec.Length;  // not 0
      double sum = 0.0;
      for (int i = 0; i < n; ++i)
        sum += vec[i];
      return sum / n;
    }

    // ------------------------------------------------------------------------

    public static double VecDot(double[] v1, double[] v2)
    {
      int n = v1.Length;
      double sum = 0.0;
      for (int i = 0; i < n; ++i)
        sum += v1[i] * v2[i];
      return sum;
    }

    // ------------------------------------------------------------------------

    public static double[] VecSubtract(double[] v1, double[] v2)
    {
      int n = v1.Length;
      double[] result = new double[n];
      for (int i = 0; i < n; ++i)
        result[i] = v1[i] - v2[i];
      return result;
    }

    // ------------------------------------------------------------------------

    public static double[] VecSelectItems(double[] vec, int[] idxs)
    {
      int n = idxs.Length;
      double[] result = new double[n];
      for (int i = 0; i < n; ++i)
      {
        result[i] = vec[idxs[i]];
      }
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

    public static void MatShow(double[][] m, int dec, int wid)
    {
      int nRows = m.Length; int nCols = m[0].Length;
      double small = 1.0 / Math.Pow(10, dec);
      for (int i = 0; i < nRows; ++i)
      {
        for (int j = 0; j < nCols; ++j)
        {
          double v = m[i][j];
          if (Math.Abs(v) < small) v = 0.0;
          Console.Write(v.ToString("F" + dec).PadLeft(wid));
        }
        Console.WriteLine("");
      }
    }

    // ------------------------------------------------------------------------

    public static void VecShow(double[] vec, int dec, int wid)
    {
      for (int i = 0; i < vec.Length; ++i)
        Console.Write(vec[i].ToString("F" + dec).PadLeft(wid));
      Console.WriteLine("");
    }

  } // class Utils

  // ==========================================================================

} // ns