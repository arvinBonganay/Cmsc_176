package logisticRegression;

import Jama.Matrix;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.*;

public class logisticRegression {
    static List<Matrix> input;
    static List<String> output;
    static Matrix theta;
    static Matrix engineeredX;
    static Matrix y;
    static int feature;
    
    public static void main(String[] args) {
        init("others/irisflowers.csv");
//        y.print(2, 2);
        scaleFeature();
//        for (Matrix x: input){
//            x.print(2,5);
//        }
        engineeredX = engineerPolynomials(2);
        sigmoid(h()).print(2, 2);
//        theta.print(2, 2);
//        engineeredX.print(2, 5);
//        Matrix m = h();
//        m.print(2, 2);
//        Matrix s = sigmoid(m);
//        s.print(2, 2);
    }

    public static void init(String filename){
        input = new ArrayList<>();
        output = new ArrayList<>();
        load_data(filename);
        feature = input.get(0).getColumnDimension();
        setOutput();
    }

    public static void load_data(String filename){
        try (FileInputStream in = new FileInputStream(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(in));) {

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
            in.close();
            reader.close();
        } catch (Exception e){
            System.out.print(e.getMessage());
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
        theta = new Matrix(y.getColumnDimension(), colSize, 0.1);
        double[][] d = new double[input.size()][colSize];
        for (int i = 0; i < colSize; i++){
            int x = Integer.valueOf(Integer.toString(i, degree + 1));
            double[] exp = new double[feature];
            for (int j = feature - 1; j >= 0; j--){
                exp[j] = (x % 10.0);
                x /= 10;
            }
//            for (int q = 0; q < feature; q++){
//                System.out.print(exp[q] +" ");
//            }
//            System.out.println("");
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
    
    public static Matrix h(){
        return engineeredX.times(theta.transpose());
    }
    
    public static Matrix sigmoid(Matrix h){
        Matrix result = new Matrix(h.getRowDimension(), h.getColumnDimension());
        for (int j = 0; j < h.getColumnDimension(); j++){
            for(int i = 0; i < h.getRowDimension(); i++){
                double x = Math.pow(Math.E, -(h.get(i, j)));
                double value = (1.0 / (1 + x));
                result.set(i, j, value);
            }
        }
        return result;
    }
    
    public static void setOutput(){
        Set<String> u = new HashSet<>(output);
        List<String> unique = new ArrayList<>(u);
        System.out.println(unique);
        y = new Matrix(input.size(), unique.size(), 0);
        int index = 0;
        for (String o: output){
            y.set(index++, unique.indexOf(o), 1);
        }
    }
}
