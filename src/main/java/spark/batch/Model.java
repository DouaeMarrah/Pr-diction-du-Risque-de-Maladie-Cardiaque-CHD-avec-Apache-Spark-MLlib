package spark.batch;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
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
public class Model {
           private static SparkSession spark = SparkSession.builder()
            .appName("DataLoader")
            .config("spark.master", "local")
            .getOrCreate();

    public static CrossValidatorModel InitialiseyModel(Dataset<Row> trainData) {
        // Assemblage des colonnes en une seule colonne de vecteurs de caracteristiques
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"age", "education", "sex", "is_smoking", "cigsPerDay", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"})
                .setOutputCol("features");

        // Mise à l'echelle des caracteristiques
        StandardScaler scaler = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .setWithStd(true)
                .setWithMean(true);

        // Creation du modèle de regression logistique
        LogisticRegression lr = new LogisticRegression()
                .setLabelCol("TenYearCHD")
                .setFeaturesCol("scaledFeatures")
                .setMaxIter(10000);

        // Creation d'un pipeline pour enchaîner les etapes de pretraitement et le modèle de regression logistique
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{assembler, scaler, lr});

        // Paramètres de la validation croisee
        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(lr.regParam(), new double[]{0.1, 0.01})
                .addGrid(lr.elasticNetParam(), new double[]{0.0, 0.5, 1.0})
                .build();

        // Creation de l'evaluateur pour la validation croisee
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("TenYearCHD")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
 
        // Initialisation de la validation croisee
        CrossValidator cv = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(5); // Nombre de plis pour la validation croisee

        // Entraînement du modèle avec la validation croisee
        CrossValidatorModel cvModel = cv.fit(trainData);

        return cvModel;
    }

    public static void TestModel(Dataset<Row> trainData) {

        // Division des donnees en ensembles d'entrainement et de test
        Dataset<Row>[] splits = trainData.randomSplit(new double[]{0.7, 0.3}, 520);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testingData = splits[1];

        //Initialisation du Model
        CrossValidatorModel model = InitialiseyModel(trainingData);

        // Predire les resultats sur les donnees de test
        Dataset<Row> predictions = model.transform(testingData);

        // Creation d'un evaluateur pour la classification
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("TenYearCHD")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        // Calcul de l'exactitude
        double accuracy = evaluator.evaluate(predictions);

        // Affichage de l'exactitude
        System.out.println("Accuracy: " + accuracy);

    }


    public static Dataset<Row> ResultatsModel(Dataset<Row> trainData,Dataset<Row> testData,String outputFilePath) {

        // Model
        CrossValidatorModel model = InitialiseyModel(trainData);
        // Predire les resultats sur les donnees de test
        Dataset<Row> predictions = model.transform(testData);

        // Selectionner toutes les colonnes sauf "features" et "scaledFeatures"
        Dataset<Row> selectedPredictions = predictions
                .select("id", "age", "education", "sex", "is_smoking", "cigsPerDay", "BPMeds", "prevalentStroke",
                        "prevalentHyp", "diabetes", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose", "prediction");

        // Enregistrer les predictions dans un fichier de sortie au format CSV
        selectedPredictions.coalesce(1)  // Pour n'avoir qu'un seul fichier de sortie
                .write()
                .option("header", true)
                .csv(outputFilePath);

        // Afficher les predictions
        selectedPredictions.show();

        // Grouper par la colonne de prediction et compter le nombre d'occurrences de chaque valeur
        Dataset<Row> predictionCounts = predictions.groupBy("prediction")
                .count()
                .withColumnRenamed("count", "prediction_count");

        // Afficher les resultats
        predictionCounts.show();

        return predictions;
    }
    
public  void AnalysePredictions(Dataset<Row> predictions){
    // Creation du DataFrame avec le nombre de predictions 0 et 1 par âge
    Dataset<Row> ageCounts = predictions.groupBy("age").agg(
            sum(when(col("prediction").equalTo(1), 1).otherwise(0)).as("prediction_1"),
            sum(when(col("prediction").equalTo(0), 1).otherwise(0)).as("prediction_0"),
            count("age").as("Total")
    );
        // Tri du DataFrame par âge
        ageCounts = ageCounts.orderBy("age");
        // Affichage du DataFrame
        ageCounts.show(30);

        // Creation du dataset pour le graphique
        DefaultCategoryDataset ageDataset = new DefaultCategoryDataset();
        List<Row> ageRows = ageCounts.collectAsList();
        for (Row row : ageRows) {
                ageDataset.addValue(row.getLong(1), "Prediction_1", row.get(0).toString());
                ageDataset.addValue(row.getLong(2), "Prediction_0", row.get(0).toString());
        }

        // Creation du graphique
        JFreeChart ageChart = ChartFactory.createBarChart(
                "Prediction distribution by age", "Age", "Number of predictions", ageDataset
        );

        // Affichage du graphique dans une fenêtre Swing
        ChartPanel ageChartPanel = new ChartPanel(ageChart);
        JFrame ageFrame = new JFrame("Prediction distribution by age");
        ageFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        ageFrame.add(ageChartPanel);
        ageFrame.pack();
        ageFrame.setVisible(true);


    // Creation du DataFrame avec le nombre de predictions 0 et 1 par sexe
    Dataset<Row> sexCounts = predictions.groupBy("sex").agg(
            sum(when(col("prediction").equalTo(1), 1).otherwise(0)).as("prediction_1"),
            sum(when(col("prediction").equalTo(0), 1).otherwise(0)).as("prediction_0"),
            count("sex").as("Total")
    );

// Affichage du DataFrame
    sexCounts.show(30);

// Creation du dataset pour le graphique avec des libelles "M" et "F"
    DefaultCategoryDataset sexDataset = new DefaultCategoryDataset();
    List<Row> sexRows = sexCounts.collectAsList();
    for (Row row : sexRows) {
        String sexLabel;
        int sexValue = row.getInt(0);
        if (sexValue == 1) {
            sexLabel = "M";
        } else {
            sexLabel = "F";
        }
        sexDataset.addValue(row.getLong(1), "Prediction_1", sexLabel);
        sexDataset.addValue(row.getLong(2), "Prediction_0", sexLabel);
    }


    // Creation du graphique
    JFreeChart sexChart = ChartFactory.createBarChart(
            "Prediction distribution by sex", "Sex", "Number of predictions", sexDataset
    );

    // Affichage du graphique dans une fenêtre Swing
    ChartPanel sexChartPanel = new ChartPanel(sexChart);
    JFrame sexFrame = new JFrame("Prediction distribution by sex");
    sexFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    sexFrame.add(sexChartPanel);
    sexFrame.pack();
    sexFrame.setVisible(true);



// Creation du DataFrame avec le nombre de predictions 0 et 1 par is_smoking
    Dataset<Row> smokingCounts = predictions.groupBy("is_smoking").agg(
            sum(when(col("prediction").equalTo(1), 1).otherwise(0)).as("prediction_1"),
            sum(when(col("prediction").equalTo(0), 1).otherwise(0)).as("prediction_0"),
            count("is_smoking").as("Total")
    );

    // Affichage du DataFrame
    smokingCounts.show(30);
    // Tri du DataFrame par is_smoking
    smokingCounts = smokingCounts.orderBy("is_smoking");

    // Creation du dataset pour le graphique
    DefaultCategoryDataset smokingDataset = new DefaultCategoryDataset();
    List<Row> smokingRows = smokingCounts.collectAsList();
    for (Row row : smokingRows) {
        smokingDataset.addValue(row.getLong(1), "Prediction_1", row.get(0).toString());
        smokingDataset.addValue(row.getLong(2), "Prediction_0", row.get(0).toString());
    }

    // Creation du graphique
    JFreeChart smokingChart = ChartFactory.createBarChart(
            "Prediction distribution by smoking status", "Smoking Status", "Number of predictions", smokingDataset
    );

    // Affichage du graphique dans une fenêtre Swing
    ChartPanel smokingChartPanel = new ChartPanel(smokingChart);
    JFrame smokingFrame = new JFrame("Prediction distribution by smoking status");
    smokingFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    smokingFrame.add(smokingChartPanel);
    smokingFrame.pack();
    smokingFrame.setVisible(true);


    // Creation du DataFrame avec le nombre de predictions 0 et 1 par heartRate
    Dataset<Row> heartRateCounts = predictions.groupBy("heartRate").agg(
            sum(when(col("prediction").equalTo(1), 1).otherwise(0)).as("prediction_1"),
            sum(when(col("prediction").equalTo(0), 1).otherwise(0)).as("prediction_0"),
            count("heartRate").as("Total")
    );
// Tri du DataFrame par heartRate
    heartRateCounts = heartRateCounts.orderBy("heartRate");
    // Affichage du DataFrame
    heartRateCounts.show(30);

    // Creation du dataset pour le graphique
    DefaultCategoryDataset heartRateDataset = new DefaultCategoryDataset();
    List<Row> heartRateRows = heartRateCounts.collectAsList();
    for (Row row : heartRateRows) {
        heartRateDataset.addValue(row.getLong(1), "Prediction_1", row.get(0).toString());
        heartRateDataset.addValue(row.getLong(2), "Prediction_0", row.get(0).toString());
    }

    // Creation du graphique
    JFreeChart heartRateChart = ChartFactory.createBarChart(
            "Prediction distribution by heart rate", "Heart Rate", "Number of predictions", heartRateDataset
    );

    // Affichage du graphique dans une fenêtre Swing
    ChartPanel heartRateChartPanel = new ChartPanel(heartRateChart);
    JFrame heartRateFrame = new JFrame("Prediction distribution by heart rate");
    heartRateFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    heartRateFrame.add(heartRateChartPanel);
    heartRateFrame.pack();
    heartRateFrame.setVisible(true);

    // Creation du DataFrame avec le nombre de predictions 0 et 1 par cigsPerDay
    Dataset<Row> cigsPerDayCounts = predictions.groupBy("cigsPerDay").agg(
            sum(when(col("prediction").equalTo(1), 1).otherwise(0)).as("prediction_1"),
            sum(when(col("prediction").equalTo(0), 1).otherwise(0)).as("prediction_0"),
            count("cigsPerDay").as("Total")
    );
// Tri du DataFrame par âge
    cigsPerDayCounts = cigsPerDayCounts.orderBy("cigsPerDay");
    // Affichage du DataFrame
    cigsPerDayCounts.show(30);

    // Creation du dataset pour le graphique
    DefaultCategoryDataset cigsPerDayDataset = new DefaultCategoryDataset();
    List<Row> cigsPerDayRows = cigsPerDayCounts.collectAsList();
    for (Row row : cigsPerDayRows) {
        cigsPerDayDataset.addValue(row.getLong(1), "Prediction_1", row.get(0).toString());
        cigsPerDayDataset.addValue(row.getLong(2), "Prediction_0", row.get(0).toString());
    }

    // Creation du graphique
    JFreeChart cigsPerDayChart = ChartFactory.createBarChart(
            "Prediction distribution by cigsPerDay", "cigsPerDay", "Number of predictions", ageDataset
    );

    // Affichage du graphique dans une fenêtre Swing
    ChartPanel cigsPerDayChartPanel = new ChartPanel(cigsPerDayChart);
    JFrame cigsPerDayFrame = new JFrame("Prediction distribution by cigsPerDay");
    cigsPerDayFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    cigsPerDayFrame.add(cigsPerDayChartPanel);
    cigsPerDayFrame.pack();
    cigsPerDayFrame.setVisible(true);

    // Creation du DataFrame avec le nombre de predictions 0 et 1 par BPMeds
    Dataset<Row> bpmedsCounts = predictions.groupBy("BPMeds").agg(
            sum(when(col("prediction").equalTo(1), 1).otherwise(0)).as("prediction_1"),
            sum(when(col("prediction").equalTo(0), 1).otherwise(0)).as("prediction_0"),
            count("BPMeds").as("Total")
    );

    // Affichage du DataFrame
    bpmedsCounts.show(30);



    // Creation du dataset pour le graphique
    DefaultCategoryDataset bpmedsDataset = new DefaultCategoryDataset();
    List<Row> bpmedsRows = bpmedsCounts.collectAsList();
    for (Row row : bpmedsRows) {
        bpmedsDataset.addValue(row.getLong(1), "Prediction_1", row.get(0).toString());
        bpmedsDataset.addValue(row.getLong(2), "Prediction_0", row.get(0).toString());
    }

    // Creation du graphique
    JFreeChart bpmedsChart = ChartFactory.createBarChart(
            "Prediction distribution by BPMeds", "BPMeds", "Number of predictions", bpmedsDataset
    );

    // Affichage du graphique dans une fenêtre Swing
    ChartPanel bpmedsChartPanel = new ChartPanel(bpmedsChart);
    JFrame bpmedsFrame = new JFrame("Prediction distribution by BPMeds");
    bpmedsFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    bpmedsFrame.add(bpmedsChartPanel);
    bpmedsFrame.pack();
    bpmedsFrame.setVisible(true);


    spark.stop();

}

    }
