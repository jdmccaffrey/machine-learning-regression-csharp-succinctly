using System;
using System.IO;
using System.Collections.Generic;

namespace QuadraticRegression
{
  internal class QuadraticRegressionProgram
  {
    static void Main(string[] args)
    {
      Console.WriteLine("\nBegin C# quadratic regression ");

      // 1. load data
      Console.WriteLine("\nLoading synthetic train (200) and test (40) data");
      string trainFile = "C:\\VSR\\Data\\synthetic_train_200.txt";
      double[][] trainX = MatLoad(trainFile, new int[] { 0, 1, 2, 3, 4 },
        ',', "#");
      double[] trainY = MatToVec(MatLoad(trainFile,
        new int[] { 5 }, ',', "#"));

      string testFile = "C:\\VSR\\Data\\synthetic_test_40.txt";
      double[][] testX = MatLoad(testFile, new int[] { 0, 1, 2, 3, 4 },
        ',', "#");
      double[] testY = MatToVec(MatLoad(testFile,
        new int[] { 5 }, ',', "#"));
      Console.WriteLine("Done ");

      Console.WriteLine("\nFirst three train X: ");
      for (int i = 0; i < 3; ++i)
        VecShow(trainX[i], 4, 8);

      Console.WriteLine("\nFirst three train y: ");
      for (int i = 0; i < 3; ++i)
        Console.WriteLine(trainY[i].ToString("F4").
          PadLeft(8));

      // 2. create and train model
      Console.WriteLine("\nCreating quadratic regression model ");
      QuadraticRegressor model = new QuadraticRegressor();
      Console.WriteLine("Done ");

      Console.WriteLine("\nTraining with MP pinv via QR-Householder ");
      model.Train(trainX, trainY);
      Console.WriteLine("Done ");

      // 3. show model weights
      Console.WriteLine("\nModel base weights: ");
      int dim = trainX[0].Length;
      for (int i = 0; i < dim; ++i)
        Console.Write(model.weights[i].ToString("F4").PadLeft(8));
      Console.WriteLine("");

      Console.WriteLine("\nModel quadratic weights: ");
      for (int i = dim; i < dim + dim; ++i)
        Console.Write(model.weights[i].
          ToString("F4").PadLeft(8));
      Console.WriteLine("");

      Console.WriteLine("\nModel interaction weights: ");
      for (int i = dim + dim; i < model.weights.Length; ++i)
      {
        Console.Write(model.weights[i].
          ToString("F4").PadLeft(8));
        if (i > dim + dim && i % dim == 0)
          Console.WriteLine("");
      }
      Console.WriteLine("");

      Console.WriteLine("\nModel bias/intercept: " + 
        model.bias.ToString("F4").PadLeft(8));

      // 4. evaluate model
      Console.WriteLine("\nEvaluating model ");
      double accTrain = model.Accuracy(trainX, trainY, 0.05);
      Console.WriteLine("Accuracy train (within 0.05) = " +
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
      Console.WriteLine("\nPredicted y = " + predY.ToString("F4"));

      Console.WriteLine("\nEnd demo ");
      Console.ReadLine();
    } // Main

    // ----------------------------------------------------------------
    // helpers for Main(): MatLoad(), MatToVec(), VecShow()
    // ----------------------------------------------------------------

    static double[][] MatLoad(string fn, int[] usecols,
      char sep, string comment)
    {
      List<double[]> result =
        new List<double[]>();
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

  public class QuadraticRegressor
  {
    public double[] weights;  // regular + quad + interactions
    public double bias;
    private Random rnd;       // for SGD training or noise regularization

    public QuadraticRegressor(int seed = 0)
    {
      this.weights = new double[0];  // keep compiler happy
      this.bias = 0;
      this.rnd = new Random(seed);
    }

    // ------------------------------------------------------------------------

    public double Predict(double[] x)
    {
      int dim = x.Length;
      double result = 0.0;

      int p = 0; // points into this.weights
      for (int i = 0; i < dim; ++i)   // regular
        result += x[i] * this.weights[p++];

      for (int i = 0; i < dim; ++i)  // quadratic
        result += x[i] * x[i] * this.weights[p++];

      for (int i = 0; i < dim - 1; ++i)  // interactions
        for (int j = i + 1; j < dim; ++j)
          result += x[i] * x[j] * this.weights[p++];

      result += this.bias;
      return result;
    }

    //public double Predict(double[] x)
    //{
    //  int n = x.Length;
    //  double result = 0.0;

    //  // expand x with quadratic and interaction terms
    //  int nTerms = n + n + ((n * (n - 1) / 2));
    //  double[] xAugmented = new double[nTerms];
    //  int p = 0;  // points into xAugmented
    //  for (int i = 0; i < n; ++i)
    //    xAugmented[p++] = x[i]; // base terms
    //  for (int i = 0; i < n; ++i)
    //    xAugmented[p++] = x[i] * x[i]; // quadratic
    //  for (int i = 0; i < n - 1; ++i)
    //    for (int j = i + 1; j < n; ++j)
    //      xAugmented[p++] = x[i] * x[j]; // interactions

    //  // linear sum of wts * x values
    //  for (int i = 0; i < xAugmented.Length; ++i)
    //    result += this.weights[i] * xAugmented[i];
    //  result += this.bias;
    //  return result;
    //}

    // ------------------------------------------------------------------------

    public void Train(double[][] trainX, double[] trainY)
    {
      // train using MP pseudo-inverse via QR-Householder
      // no regulaization
      // w = pinv(designX) * y
      int nRows = trainX.Length; // not used this implementation
      int dim = trainX[0].Length;
      int nInteractions = (dim * (dim - 1)) / 2;
      this.weights = new double[dim + dim + nInteractions];

      double[][] Xa = MatAugment(trainX);  // add columns
      double[][] Xd = MatToDesign(Xa);      // add leading 1.0s col

      double[][] Xpinv = QRHouseholder.MatPinv(Xd);

      double[] biasAndWts = MatVecProd(Xpinv, trainY);
      this.bias = biasAndWts[0];  // bias is at [0]
      for (int i = 1; i < biasAndWts.Length; ++i)
        this.weights[i - 1] = biasAndWts[i];
      return;
    }

    // ------------------------------------------------------------------------

    private static double[][] MatAugment(double[][] trainX)
    {
      // add quadratic and interaction columns
      int nRows = trainX.Length;  // src and dest
      int dim = trainX[0].Length;  // src
      int nInteractions = dim * (dim - 1) / 2;
      int nColsDest = dim + dim + nInteractions;

      double[][] result = new double[nRows][];
      for (int i = 0; i < nRows; i++)
        result[i] = new double[nColsDest];

      for (int i = 0; i < nRows; ++i)
      {
        int p = 0; // points to column of result

        for (int j = 0; j < dim; ++j) // base
          result[i][p++] = trainX[i][j];

        for (int j = 0; j < dim; ++j) // quadratic
          result[i][p++] = trainX[i][j] * trainX[i][j];

        for (int j = 0; j < nInteractions - 1; ++j)
          for (int k = j + 1; k < dim; ++k)
            result[i][p++] = trainX[i][j] * trainX[i][k];
      }

      return result;
    }

    // ------------------------------------------------------------------------

    private static double[][] MatToDesign(double[][] X)
    {
      // add leading column of 1.0s to handle bias term
      int nRows = X.Length;  // src and dest
      int dim = X[0].Length;

      double[][] result = MatMake(nRows, dim + 1);  // extra col
      for (int i = 0; i < nRows; ++i)
      {
        result[i][0] = 1.0;
        for (int j = 1; j < result[0].Length; ++j)
        {
          result[i][j] = X[i][j - 1];
        }
      }
      return result;
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

    private static double[] MatVecProd(double[][] M, double[] v)
    {
      // helper for Train()
      int nRows = M.Length;
      int nCols = M[0].Length;
      int n = v.Length;
      if (nCols != n)
        throw new Exception("non-comform in MatVecProd");

      double[] result = new double[nRows];
      for (int i = 0; i < nRows; ++i)
        for (int k = 0; k < nCols; ++k)
          result[i] += M[i][k] * v[k];

      return result;
    }

    // ------------------------------------------------------------------------

    public double Accuracy(double[][] dataX, double[] dataY, double pctClose)
    {
      int numCorrect = 0; int numWrong = 0;
      for (int i = 0; i < dataX.Length; ++i)
      {
        double actualY = dataY[i];
        double predY = this.Predict(dataX[i]);
        if (Math.Abs(predY - actualY) < Math.Abs(pctClose * actualY))
          ++numCorrect;
        else
          ++numWrong;
      }
      return (numCorrect * 1.0) / (numWrong + numCorrect);
    }

    // ------------------------------------------------------------------------

    public double MSE(double[][] dataX, double[] dataY)
    {
      int n = dataX.Length;
      double sum = 0.0;
      for (int i = 0; i < n; ++i)
      {
        double actualY = dataY[i];
        double predY = this.Predict(dataX[i]);
        sum += (actualY - predY) * (actualY - predY);
      }
      return sum / n;
    }

  } // class QuadraticRegressor

  // ==========================================================================

  public class QRHouseholder
  {
    // container for MP pseudo-inverse via QR-Householder
    // A = Q * R
    // pinv(A) = inv(R) * inv(Q)  note order matters
    //         = inv upper tri (easy) * transpose (easy)

    public static double[][] MatPinv(double[][] M)
    {
      double[][] Q; double[][] R;
      MatDecompQR(M, out Q, out R);  // Householder
      double[][] Ri = MatInvUpperTri(R);
      double[][] Qi = MatTranspose(Q);
      double[][] result = MatProduct(Ri, Qi);
      return result;
    }

    // ------------------------------------------------------------------------

    private static void MatDecompQR(double[][] A,
      out double[][] Q, out double[][] R)
    {
      // Householder algorithm
      int m = A.Length; int n = A[0].Length;
      if (m < n)
        Console.WriteLine("FATAL: nRows must be gte nCols");

      double[][] QQ = MatMake(m, m); // working full Q
      for (int i = 0; i < m; ++i)
        QQ[i][i] = 1.0;  // identity matrix

      double[][] RR = MatMake(m, n);
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
          RR[i][j] = A[i][j]; // copy of A is working R

      int k = Math.Min(m, n);  // or just use n
      for (int j = 0; j < k; ++j) // main processing loop
      {
        int xn = m - j;
        double[] x = new double[xn];
        for (int i = 0; i < xn; ++i)
          x[i] = RR[j + i][j];

        double ss = 0.0;
        for (int i = 0; i < xn; ++i)
          ss += x[i] * x[i];
        double normX = Math.Sqrt(ss);

        // if (normX == 0.0) continue;  // risky
        if (Math.Abs(normX) < 1.0e-12) continue;

        double sign;
        if (x[0] >= 0.0) sign = -1.0;
        else sign = 1.0; // counter-intuitive

        double[] u = new double[xn];
        for (int i = 0; i < xn; ++i)
          u[i] = x[i] / (x[0] - sign * normX); // check div 0
        u[0] = 1.0;

        // compute scaling factor tau = 2 / (u^T * u)
        double tau = -sign * (x[0] - sign * normX) / normX;

        // dimensions for sub-matrices
        int nRowsSubR = m - j; int nColsSubR = n - j;
        int nRowsSubQ = m; int nColsSubQ = m - j;

        double[] vr = new double[nColsSubR];
        for (int c = 0; c < nColsSubR; ++c)
        {
          double acc = 0.0;
          for (int r = 0; r < nRowsSubR; ++r)
            acc += u[r] * RR[j + r][j + c];
          vr[c] = acc;
        }

        double[] vq = new double[nRowsSubQ];
        for (int r = 0; r < nRowsSubQ; ++r)
        {
          double acc = 0.0;
          for (int c = 0; c < nColsSubQ; ++c)
            acc += u[c] * QQ[r][j + c];
          vq[r] = acc;
        }

        // update sub-R
        for (int r = 0; r < nRowsSubR; ++r)
          for (int c = 0; c < nColsSubR; ++c)
            RR[j + r][j + c] -= tau * u[r] * vr[c];

        // update sub-Q
        for (int r = 0; r < nRowsSubQ; ++r)
          for (int c = 0; c < nColsSubQ; ++c)
            QQ[r][j + c] -= tau * vq[r] * u[c];

      } // j

      // extract QQ RR into out params
      Q = MatMake(m, n);
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
          Q[i][j] = QQ[i][j];

      R = MatMake(n, n);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          R[i][j] = RR[i][j];

      return;
    } // MatDecompQR

    // ------------------------------------------------------------------------

    public static double[][] MatInvUpperTri(double[][] U)
    {
      int n = U.Length;  // must be square matrix

      double[][] result = MatMake(n, n);
      for (int i = 0; i < n; ++i)
        result[i][i] = 1.0;
      for (int k = 0; k < n; ++k)
      {
        for (int j = 0; j < n; ++j)
        {
          for (int i = 0; i < k; ++i)
          {
            result[j][k] -= result[j][i] * U[i][k];
          }
          result[j][k] /= U[k][k];
        }
      }
      return result;
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

    private static double[][] MatTranspose(double[][] M)
    {
      int nRows = M.Length;
      int nCols = M[0].Length;
      double[][] result = MatMake(nCols, nRows);
      for (int i = 0; i < nRows; ++i)
        for (int j = 0; j < nCols; ++j)
          result[j][i] = M[i][j];
      return result;
    }

    // ------------------------------------------------------------------------

    private static double[][] MatProduct(double[][] A, double[][] B)
    {
      int aRows = A.Length; int aCols = A[0].Length;
      int bRows = B.Length; int bCols = B[0].Length;
      if (aCols != bRows)
        throw new Exception("Non-conformable matrices");

      double[][] result = new double[aRows][];
      for (int i = 0; i < aRows; ++i)
        result[i] = new double[bCols];

      for (int i = 0; i < aRows; ++i) // each row of A
        for (int j = 0; j < bCols; ++j) // each col of B
          for (int k = 0; k < aCols; ++k)
            result[i][j] += A[i][k] * B[k][j];

      return result;
    }

  } // class QRHouseholder

  // ==========================================================================

} // ns