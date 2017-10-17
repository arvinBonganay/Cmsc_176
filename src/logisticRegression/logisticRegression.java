package logisticRegression;

import Jama.Matrix;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.*;

public class logisticRegression {
    static List<Matrix> input;
    static List<String> classification;
    static Matrix theta;
    static Matrix polyX;
    static Matrix y;
    static Matrix h; // h(x())
    static int feature;
    static double lambda = 1;
    static double alpha =  1;
    static List<List<Double>> costHist = new ArrayList<>();
    
    public static void main(String[] args) {
        init("others/irisflowers.csv");
        gradientDescent(100);
        for(List<Double> a: costHist){
            System.out.println(a);
        }
    }

    public static void init(String filename){
        input = new ArrayList<>();
        load_data(filename);
        feature = input.get(0).getColumnDimension();
        scaleFeature();
        polyX = engineerPolynomials(1);
        sigmoid(hOfX());
    }

    public static void load_data(String filename){
        try (FileInputStream in = new FileInputStream(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(in));) {
            List<String> output = new ArrayList<>();
            String line;
            line = reader.readLine();// for the column names
            while ((line = reader.readLine()) != null){
                StringTokenizer st = new StringTokenizer(line, ",");
                int i = 0;
                double[][] x = new double[1][st.countTokens() - 1];
                while (st.hasMoreElements()){
                    String token = (st.nextToken().trim());
                    if (st.hasMoreElements()){
                        x[0][i++] = Double.valueOf(token); 
                    } else {
                        output.add(token);
                    }
                }
                input.add(new Matrix(x));
            }
            setOutput(output);
            in.close();
            reader.close();
        } catch (Exception e){
            System.out.print(e.getMessage());
        }
    }
    
    public static void setOutput(List<String> output){
        Set<String> u = new HashSet<>(output);
        List<String> unique = new ArrayList<>(u);
        classification = new ArrayList<>(unique);
        y = new Matrix(input.size(), unique.size(), 0);
        int index = 0;
        for (String o: output){
            y.set(index++, unique.indexOf(o), 1);
        }
    }
    
    public static void scaleFeature(){
        for (int i = 0; i < feature; i++){
            double m = mean(i);
            double s = standardDev(i, m);
            for (Matrix x: input){
                double value = (x.get(0, i) - m) / s;
                x.set(0, i, value);
            }
        }
    }

    public static double mean(int feature){
        double sum = 0.0;
        for (Matrix x: input){
            sum += x.get(0, feature);
        }
        return sum / input.size();
    }
    
    public static double standardDev(int feature, double mean){
        double sum = 0.0;
        for (Matrix x:input){
            sum += Math.pow(x.get(0, feature) - mean, 2);
        }
        return Math.sqrt(sum / input.size());
    }
    
    public static Matrix engineerPolynomials(int degree){
        int colSize = (int)Math.pow(degree + 1, feature);
        theta = new Matrix(y.getColumnDimension(), colSize, 1);
        double[][] d = new double[input.size()][colSize];
        for (int i = 0; i < colSize; i++){
            int x = Integer.valueOf(Integer.toString(i, degree + 1));
            double[] exp = new double[feature];
            for (int j = feature - 1; j >= 0; j--){
                exp[j] = (x % 10.0);
                x /= 10;
            }
            int index = 0;
            for (Matrix matrix:input){
                double value = 1;
                for (int e = 0; e < exp.length; e++){
                    value *= Math.pow(matrix.get(0, e), exp[e]);
                }
                d[index++][i] = value;
            }
        }
        return new Matrix(d);
    }
    
    public static Matrix hOfX(){
        return polyX.times(theta.transpose());
    }
    
    public static void sigmoid(Matrix x){
        Matrix result = new Matrix(x.getRowDimension(), x.getColumnDimension());
        for (int j = 0; j < x.getColumnDimension(); j++){
            for(int i = 0; i < x.getRowDimension(); i++){
                double a = Math.pow(Math.E, -(x.get(i, j)));
                double value = (1.0 / (1 + a));
//                if (value >= 1.0){
//                    value = Math.nextAfter(1.0, 0);   // so that there would be no log(0)
//                }
                result.set(i, j, value);
            }
        }
        h = result;
    }
    
    public static double cost(int index){
      
        Matrix a = h.getMatrix(0, h.getRowDimension() - 1, index, index); // submatrix of g(h(x))
        Matrix a1 = a.copy();  // log( g() )
        Matrix a2 = a.copy();  // log( 1 - g() ) 
        for (int i = 0; i < a.getRowDimension(); i++){
            a1.set(i, 0, Math.log10(a.get(i, 0)));      
            a2.set(i, 0, Math.log10(1 - a.get(i, 0)));  
        }
        Matrix b1 = y.getMatrix(0, y.getRowDimension() - 1, index, index).transpose();              // y
        Matrix b2 = (new Matrix(b1.getRowDimension(), b1.getColumnDimension(), 1)).minusEquals(b1); // 1 - y 
        double result = -1.0 / a.getRowDimension() * (b1.times(a1).get(0, 0) + b2.times(a2).get(0, 0));
        
        Matrix c = theta.getMatrix(index, index, 0, theta.getColumnDimension() - 1); // theta submatrix 
        double thetaSum = c.times(c.transpose()).get(0, 0);                          // summation of theta ^ 2
        result += lambda / (2 * a.getRowDimension()) * thetaSum;
        return result;
    }
    
    public static void gradientDescent(int iter){
        
        while (iter >= 0){
            for (int i = 0; i < theta.getRowDimension(); i++){
                Matrix a = h.getMatrix(0, h.getRowDimension() - 1, i, i);
                Matrix b = y.getMatrix(0, y.getRowDimension() - 1, i, i);
                Matrix c = a.minus(b);
                Matrix m = theta.getMatrix(i, i, 0, theta.getColumnDimension() - 1); 
                double thetaSum = m.times(new Matrix(m.getRowDimension(), m.getColumnDimension(), 1).transpose()).get(0, 0); 
                thetaSum  = lambda / a.getRowDimension() * thetaSum;
                for (int j = 0; j < theta.getColumnDimension(); j++){
                    double d = c.transpose().times(polyX.getMatrix(0, polyX.getRowDimension() - 1, j, j)).get(0, 0);
                    double value = theta.get(i, j) - alpha * 1 / a.getRowDimension() * d + thetaSum;
                    theta.set(i, j, value);
                }
            }
            List<Double> cost = new ArrayList<>();
            cost.add(cost(0));
            cost.add(cost(1));
            cost.add(cost(2));
            costHist.add(cost);
            iter--;
        }
    }
}
