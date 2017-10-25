package logisticRegression;

import Jama.Matrix;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.*;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;

public class logisticRegression {
    static List<Matrix> input;
    static List<String> classification;
    static Matrix theta;
    static Matrix polyX;
    static Matrix y;
    static int feature;
    static double lambda = 0.001;
    static double alpha =  0.01;
    static List<List<Double>> costHist = new ArrayList<>();
    
    public static void main(String[] args) {
        int degree = 2;
        int iterations = 200;
        
        input = new ArrayList<>();
        load_data("others/irisflowers.csv");
        feature = input.get(0).getColumnDimension();
        scaleFeature();
        polyX = engineerPolynomials(degree);
        gradientDescent(iterations);
        graph();
        Matrix h = sigmoid(hOfX());
        System.out.println(classification.get(0) + " final cost: " + cost(0, h));
        System.out.println(classification.get(1) + " final cost: " + cost(1, h));
        System.out.println(classification.get(2) + " final cost: " + cost(2, h));

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
        classification = new ArrayList<>();
        for (String s: output){
            if (!classification.contains(s)){
                classification.add(s);
            }
        }
        y = new Matrix(input.size(), classification.size(), 0);
        int index = 0;
        for (String o: output){
            y.set(index++, classification.indexOf(o), 1);
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
    
    public static Matrix sigmoid(Matrix x){
        Matrix result = new Matrix(x.getRowDimension(), x.getColumnDimension());
        for (int j = 0; j < x.getColumnDimension(); j++){
            for(int i = 0; i < x.getRowDimension(); i++){
                double a = Math.pow(Math.E, -(x.get(i, j)));
                double value = (1.0 / (1 + a));
                if (value >= 1.0){
                    value = Math.nextAfter(1.0, 0);   // so that there would be no log(0)
                }
                result.set(i, j, value);
            }
        }
        return result;
    }
    
    public static double cost(int index, Matrix h){
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
        Matrix h = sigmoid(hOfX());
        while (iter >= 0){
            List<Double> cost = new ArrayList<>();
            Matrix newH = sigmoid(hOfX());
            cost.add(cost(0, newH));
            cost.add(cost(1, newH));
            cost.add(cost(2, newH));
            costHist.add(cost);
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
            iter--;
        }
    }
    
    public static void graph(){
        ApplicationFrame app = new ApplicationFrame("Cost vs Iteration");
        XYSeriesCollection data = new XYSeriesCollection();
        for (String s: classification){
            XYSeries dataset = new XYSeries(s);
            int index = classification.indexOf(s);
            for (int i = 0; i < costHist.size(); i++){
                dataset.add(i+1, costHist.get(i).get(index));
            }
            data.addSeries(dataset);
        }
        JFreeChart chart = ChartFactory.createXYLineChart("Cost vs iteration", "Iteration", "Cost", data,
                                                  PlotOrientation.VERTICAL,
                                                  true, true, false);

        ChartPanel chartPanel = new ChartPanel( chart );
        chartPanel.setPreferredSize( new java.awt.Dimension( 600 , 367 ) );
        XYPlot plot = chart.getXYPlot( );
      
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer( );
        plot.setRenderer( renderer ); 
        app.setContentPane( chartPanel );
        app.pack();
        RefineryUtilities.centerFrameOnScreen(app);
        app.setVisible(true);
    }
}
