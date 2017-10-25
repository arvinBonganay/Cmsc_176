package linearRegression;

import java.io.*;
import java.util.*;
import Jama.Matrix;
import org.jfree.chart.*;
import org.jfree.ui.*;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.*;

public class linearRegression {
    static List<Matrix> input;
    static List<Double> output;
    static Matrix theta;
    static List<Double> costHist;


    public static void main(String[] args) {
        init("others/HousePricingRelationship.in");
        double n = alphaValue();
        System.out.println(n);
        gradientDescent(n, 100);
        graph();
    }

    public static void init(String filename){
        input = new ArrayList<>();
        output = new ArrayList<>();

        load_data(filename);

        int n = input.get(0).getColumnDimension();
        double[][] t = new double[1][n];
        for (int i = 0; i < n; i++){
            t[0][i] = 1;
        }
        theta = new Matrix(t);
        costHist = new ArrayList<>();


    }

    public static void load_data(String filename){
        try (FileInputStream in = new FileInputStream(filename);
            BufferedReader reader = new BufferedReader(new InputStreamReader(in));) {

            String line;
            while ((line = reader.readLine()) != null){
                StringTokenizer st =   new StringTokenizer(line, ",");
                int i = 0;
                int n = st.countTokens();
                double[][] x = new double[1][n];
                x[0][0] = 1;
                while (st.hasMoreElements()){
                    Double token = new Double(st.nextToken().trim());
                    if (st.hasMoreElements()){
                        x[0][++i] = token;
                    } else {
                        output.add(token);
                    }
                }
                input.add(new Matrix(x));
            }
            in.close();
            reader.close();
        } catch (Exception e){
            System.out.print("File does not exist");
        }
    }

    // h(x)
    public static double hValue(Matrix x, Matrix theta){
        return x.times(theta.transpose()).get(0, 0);
    }

    public static double cost(List<Matrix> x, List<Double> y, Matrix theta){
        double sum = 0;
        for (int i = 0; i < x.size(); i++){
            double z = hValue(x.get(i), theta) - y.get(i);
            sum += Math.pow(z, 2);
        }
        return (sum / (2 * x.size()));
    }

    public static double partialDerivative(List<Matrix> x, List<Double> y, Matrix theta, int n){
        double sum = 0;
        for (int i = 0; i < x.size(); i++){
            double h = hValue(x.get(i), theta);
            double z = (h - y.get(i)) * x.get(i).get(0, n);
            sum += z;
        }
        return sum / (x.size());
    }

    public static double alphaValue(){
        double alpha = 1;
        Matrix t = theta.copy();
        double currentCost = cost(input, output, t);
        while (true){
            updateTheta(alpha, t);
            double newCost = cost(input, output, t);
            if(newCost < currentCost){
                break;
            } else {
                alpha = alpha * 0.1;
                t = theta.copy();
            }
        }
        return alpha;
    }

    public static void updateTheta(double alpha, Matrix t){
        List<Double> temp = new ArrayList<>();
        for (int n = 0; n < theta.getColumnDimension(); n++){
                temp.add(theta.get(0, n) -(alpha * partialDerivative(input, output, t, n)));
            }
        for (int i = 0; i < temp.size(); i++){
            t.set(0, i, temp.get(i));
        }
    }

    public static void gradientDescent(double alpha, int iter){
        while(iter > 0){
            costHist.add(cost(input, output, theta));
            updateTheta(alpha, theta);
            iter--;
        }
    }

    public static void graph(){
        ApplicationFrame app = new ApplicationFrame("Cost vs Iteration");

        XYSeries dataset = new XYSeries("cost");
        for (int i = 0; i < costHist.size(); i++){
            dataset.add(i+ 1, (double)costHist.get(i) );
        }
        XYSeriesCollection data = new XYSeriesCollection(dataset);
        JFreeChart chart = ChartFactory.createXYLineChart("Cost vs iteration", "Iteration", "Cost", data,
                                                  PlotOrientation.VERTICAL,
                                                  true, true, false);

        ChartPanel chartPanel = new ChartPanel( chart );
        chartPanel.setPreferredSize( new java.awt.Dimension( 600 , 367 ) );
        app.setContentPane( chartPanel );
        app.pack();
        RefineryUtilities.centerFrameOnScreen(app);
        app.setVisible(true);
    }
}
