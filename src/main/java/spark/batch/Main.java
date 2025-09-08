package spark.batch;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.VectorAssembler;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.category.DefaultCategoryDataset;
import java.awt.Dimension;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import javax.swing.*;
import java.util.List; 
import static org.apache.spark.sql.functions.*;

public class Main {
    public static void main(String[] args) {

        SparkSession spark = SparkSession.builder()
                .appName("DataLoader")
                .config("spark.master", "local")
                .getOrCreate();
        // Vérifier que les arguments sont fournis
        if (args.length < 3) {
            System.err.println("Usage: CHDPredictionTask <trainDataPath> <testDataPath> <outputFilePath>");
            System.exit(1);
        }
        Dataset<Row>[] datasets = Data.loadTrainTestData(args[0], args[1]);
        Dataset<Row> trainData = datasets[0];
        Dataset<Row> testData = datasets[1];

        Dataset<Row> trainData2 = Data.prepare(trainData);

    Data.analyse(trainData2);

    Model model=new Model();
        model.AccuracyModel(trainData2);
        Dataset<Row> testData2 = Data.prepare(testData);
        Dataset<Row>  predictions=model.TestModel(trainData2,testData2,args[2]);
       model.AnalysePredictions(predictions);

    }
}
