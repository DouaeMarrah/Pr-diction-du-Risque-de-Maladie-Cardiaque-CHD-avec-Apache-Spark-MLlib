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
public class Data {
    private static SparkSession spark = SparkSession.builder()
            .appName("DataLoader")
            .config("spark.master", "local")
            .getOrCreate();

    public static Dataset<Row>[] loadTrainTestData(String trainDataPath, String testDataPath) {

        Dataset<Row> trainData = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv(trainDataPath);

        Dataset<Row> testData = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv(testDataPath);

        return new Dataset[]{trainData, testData};
    }

    public static Dataset<Row> prepare(Dataset<Row> Data) {
        // Afficher le schéma des données de train
        Data.printSchema();
        // Visualiser la distribution des données de chaque colonne
        Data.describe().show();

        // Remplacer les valeurs "M" par 1 et "F" par 0 dans la colonne "sex"
        Data = Data.withColumn("sex", when(col("sex").equalTo("M"), 1).otherwise(0));
        // Remplacer les valeurs "Yes" par 1 et "No" par 0 dans la colonne "is_smoking"
        Data = Data.withColumn("is_smoking", when(col("is_smoking").equalTo("YES"), 1).otherwise(0));

        // Liste des noms de colonnes dans Data
        String[] columnNames = Data.columns();
        // Liste pour stocker le nombre de données manquantes pour chaque colonne
        long[] missingCount = new long[columnNames.length];

        // Calculer le nombre de données manquantes pour chaque colonne
        for (int i = 0; i < columnNames.length; i++) {
            missingCount[i] = Data.filter(col(columnNames[i]).isNull()).count();
        }

        // Afficher chaque colonne et le nombre de données manquantes
        System.out.println("Nombre de données manquantes par colonne :");
        System.out.println("----------------------------------------");
        for (int i = 0; i < columnNames.length; i++) {
            String columnName = columnNames[i];
            long count = missingCount[i];
            System.out.printf("%-20s : %-10d\n", columnName, count);
        }

// Remplacer les valeurs manquantes par la médiane de chaque colonne
        for (String colName : Data.columns()) {
            double median = Data.stat().approxQuantile(colName, new double[]{0.5}, 0.0)[0];
            Data = Data.na().fill(median, new String[]{colName});
        }

        // Créer un ensemble de données pour le graphique
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        for (int i = 0; i < columnNames.length; i++) {
            dataset.addValue(missingCount[i], "Missing Data", columnNames[i]);
        }

        // Créer le graphique
        JFreeChart barChart = ChartFactory.createBarChart(
                "Missing Data per Column", "Columns", "Missing Data Count",
                dataset, PlotOrientation.VERTICAL, false, true, false);

        // Afficher le graphique dans une fenêtre Swing
        ChartPanel chartPanel = new ChartPanel(barChart);
        JFrame frame = new JFrame("Missing Data per Column");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(chartPanel);
        frame.pack();
        frame.setVisible(true);


        

        // Afficher les premières lignes du DataFrame pour vérification
        Data.show(30);
        // Liste pour stocker le nombre de données manquantes pour chaque colonne après le remplacement
        long[] missingCountAfter = new long[columnNames.length];

        // Calculer le nombre de données manquantes pour chaque colonne après le remplacement
        for (int i = 0; i < columnNames.length; i++) {
            missingCountAfter[i] = Data.filter(col(columnNames[i]).isNull()).count();
        }

        // Créer un ensemble de données pour le deuxième graphique
        DefaultCategoryDataset datasetAfter = new DefaultCategoryDataset();
        for (int i = 0; i < columnNames.length; i++) {
            datasetAfter.addValue(missingCountAfter[i], "Missing Data (After)", columnNames[i]);
        }

        // Créer le deuxième graphique
        JFreeChart barChartAfter = ChartFactory.createBarChart(
                "Missing Data per Column (After)", "Columns", "Missing Data Count",
                datasetAfter, PlotOrientation.VERTICAL, false, true, false);

        // Afficher le deuxième graphique dans une fenêtre Swing
        ChartPanel chartPanelAfter = new ChartPanel(barChartAfter);
        JFrame frameAfter = new JFrame("Missing Data per Column (After)");
        frameAfter.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frameAfter.add(chartPanelAfter);
        frameAfter.pack();
        frameAfter.setVisible(true);

        // Visualiser la distribution des données de chaque colonne
        Data.describe().show();

        return Data;
    }

    public static void analyse(Dataset<Row> Data) {
        // Supprimer la colonne "id"
        Data = Data.drop("id");

        // Création du DataFrame avec le nombre de 'TenYearCHD' et 'Non TenYearCHD' par âge
        Dataset<Row> ageCounts = Data.groupBy("age").agg(
                sum(when(col("TenYearCHD").equalTo(1), 1).otherwise(0)).as("TenYearCHD"),
                count("age").as("Total")
        );
         
        ageCounts = ageCounts.orderBy("age"); // Tri du DataFrame par âge
        ageCounts.show(30); // Affichage du DataFrame

        // Création du dataset pour le graphique
        DefaultCategoryDataset ageDataset = new DefaultCategoryDataset();
        List<Row> ageRows = ageCounts.collectAsList();
        for (Row row : ageRows) {
            ageDataset.addValue(row.getLong(1), "TenYearCHD", row.get(0).toString());
            ageDataset.addValue(row.getLong(2) - row.getLong(1), "Non TenYearCHD", row.get(0).toString());
        }

        // Création du graphique
        JFreeChart ageChart = ChartFactory.createBarChart(
                "Age distribution with Ten years CHD", "Age", "Number of patients", ageDataset
        );

        // Affichage du graphique dans une fenêtre Swing
        ChartPanel ageChartPanel = new ChartPanel(ageChart);
        JFrame ageFrame = new JFrame("Age distribution with Ten years CHD");
        ageFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        ageFrame.add(ageChartPanel);
        ageFrame.pack();
        ageFrame.setVisible(true);


        // Création du DataFrame avec le nombre de 'TenYearCHD' et 'Non TenYearCHD' par sexe
        Dataset<Row> genderCounts = Data.groupBy("sex").agg(
                sum(when(col("TenYearCHD").equalTo(1), 1).otherwise(0)).as("TenYearCHD"),
                count("sex").as("Total")
        );

        // Ajout d'une colonne "GenderLabel" avec des libellés "M" et "F" basés sur la colonne "sex"
        Dataset<Row> genderCountsWithLabels = genderCounts.withColumn("GenderLabel",
                when(col("sex").equalTo(1), "M").otherwise("F")
        ); 

// Affichage du DataFrame avec les libellés "M" et "F"
        genderCountsWithLabels.show(30);


        // Création du dataset pour le graphique
        DefaultCategoryDataset genderDataset = new DefaultCategoryDataset();
        List<Row> genderRows = genderCountsWithLabels.collectAsList();
        for (Row row : genderRows) {
            String genderLabel = row.getString(3);  // Utiliser getString() pour récupérer la colonne "GenderLabel"
            genderDataset.addValue(row.getLong(1), "TenYearCHD", genderLabel);
            genderDataset.addValue(row.getLong(2) - row.getLong(1), "Non TenYearCHD", genderLabel);
        }


        // Création du graphique
        JFreeChart genderChart = ChartFactory.createBarChart(
                "Sex distribution with Ten years CHD", "Sex", "Number of patients", genderDataset
        );


        // Affichage du graphique dans une fenêtre Swing
        ChartPanel genderChartPanel = new ChartPanel(genderChart);
        JFrame genderFrame = new JFrame("Sex distribution with Ten years CHD");
        genderFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        genderFrame.add(genderChartPanel);
        genderFrame.pack();
        genderFrame.setVisible(true);

// Création du DataFrame avec le nombre de 'TenYearCHD' et 'Non TenYearCHD' par is_smoking
        Dataset<Row> smokingCounts = Data.groupBy("is_smoking").agg(
                sum(when(col("TenYearCHD").equalTo(1), 1).otherwise(0)).as("TenYearCHD"),
                count("is_smoking").as("Total")
        );

// Tri du DataFrame par is_smoking
        smokingCounts = smokingCounts.orderBy("is_smoking");

// Affichage du DataFrame
        smokingCounts.show(30);

// Création du dataset pour le graphique avec des libellés "YES" et "NO"
        DefaultCategoryDataset smokingDataset = new DefaultCategoryDataset();
        List<Row> smokingRows = smokingCounts.collectAsList();
        for (Row row : smokingRows) {
            String smokingLabel;
            int smokingValue = row.getInt(0);
            if (smokingValue == 1) {
                smokingLabel = "YES";
            } else {
                smokingLabel = "NO";
            }
            smokingDataset.addValue(row.getLong(1), "TenYearCHD", smokingLabel);
            smokingDataset.addValue(row.getLong(2) - row.getLong(1), "Non TenYearCHD", smokingLabel);
        }


        // Création du graphique
        JFreeChart smokingChart = ChartFactory.createBarChart(
                "is_smoking distribution with Ten years CHD", "is_smoking", "Number of patients", smokingDataset
        );

        // Affichage du graphique dans une fenêtre Swing
        ChartPanel smokingChartPanel = new ChartPanel(smokingChart);
        JFrame smokingFrame = new JFrame("is_smoking distribution with Ten years CHD");
        smokingFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        smokingFrame.add(smokingChartPanel);
        smokingFrame.pack();
        smokingFrame.setVisible(true);

        // Création du DataFrame avec le nombre de 'TenYearCHD' et 'Non TenYearCHD' par 'heartRate'
        Dataset<Row> heartRateCounts = Data.groupBy("heartRate").agg(
                sum(when(col("TenYearCHD").equalTo(1), 1).otherwise(0)).as("TenYearCHD"),
                count("heartRate").as("Total")
        );

        // Tri du DataFrame par heartRate
        heartRateCounts = heartRateCounts.orderBy("heartRate");
        // Affichage du DataFrame
        heartRateCounts.show(30);

        // Création du dataset pour le graphique
        DefaultCategoryDataset heartRateDataset = new DefaultCategoryDataset();
        List<Row> heartRateRows = heartRateCounts.collectAsList();
        for (Row row : heartRateRows) {
            heartRateDataset.addValue(row.getLong(1), "TenYearCHD", row.get(0).toString());
            heartRateDataset.addValue(row.getLong(2) - row.getLong(1), "Non TenYearCHD", row.get(0).toString());
        }

        // Création du graphique
        JFreeChart heartRateChart = ChartFactory.createBarChart(
                "heartRate distribution with Ten years CHD", "heartRate", "Number of patients", heartRateDataset
        );

        // Affichage du graphique dans une fenêtre Swing
        ChartPanel heartRateChartPanel = new ChartPanel(heartRateChart);
        JFrame heartRateFrame = new JFrame("heartRate distribution with Ten years CHD");
        heartRateFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        heartRateFrame.add(heartRateChartPanel);
        heartRateFrame.pack();
        heartRateFrame.setVisible(true);



        // Création du DataFrame avec le nombre de 'TenYearCHD' et 'Non TenYearCHD' par cigsPerDay
        Dataset<Row> cigsPerDayCounts = Data.groupBy("cigsPerDay").agg(
                sum(when(col("TenYearCHD").equalTo(1), 1).otherwise(0)).as("TenYearCHD"),
                count("cigsPerDay").as("Total")
        );

        // Tri du DataFrame par cigsPerDay
        cigsPerDayCounts = cigsPerDayCounts.orderBy("cigsPerDay");

// Affichage du DataFrame
        cigsPerDayCounts.show(30);

// Création du dataset pour le graphique
        DefaultCategoryDataset cigsPerDayDataset = new DefaultCategoryDataset();
        List<Row> cigsPerDayRows = cigsPerDayCounts.collectAsList();
        for (Row row : cigsPerDayRows) {
            cigsPerDayDataset.addValue(row.getLong(1), "TenYearCHD", row.get(0).toString());
            cigsPerDayDataset.addValue(row.getLong(2) - row.getLong(1), "Non TenYearCHD", row.get(0).toString());
        }

// Création du graphique
        JFreeChart cigsPerDayChart = ChartFactory.createBarChart(
                "Cigs Per Day distribution with Ten years CHD", "Cigs Per Day", "Number of patients", cigsPerDayDataset
        );

// Affichage du graphique dans une fenêtre Swing
        ChartPanel cigsPerDayChartPanel = new ChartPanel(cigsPerDayChart);
        JFrame cigsPerDayFrame = new JFrame("Cigs Per Day distribution with Ten years CHD");
        cigsPerDayFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        cigsPerDayFrame.add(cigsPerDayChartPanel);
        cigsPerDayFrame.pack();
        cigsPerDayFrame.setVisible(true);


// Création du DataFrame avec le nombre de 'TenYearCHD' et 'Non TenYearCHD' par 'BPMeds'
        Dataset<Row> chdCounts = Data.groupBy("BPMeds").agg(
                sum(when(col("TenYearCHD").equalTo(1), 1).otherwise(0)).as("TenYearCHD"),
                count("BPMeds").as("Total")
        );

        // Tri du DataFrame par BPMeds
        chdCounts = chdCounts.orderBy("BPMeds");
// Affichage du DataFrame
        chdCounts.show(30);

// Création du dataset pour le graphique
        DefaultCategoryDataset chdDataset = new DefaultCategoryDataset();
        List<Row> chdRows = chdCounts.collectAsList();
        for (Row row : chdRows) {
            chdDataset.addValue(row.getLong(1), "TenYearCHD", row.get(0).toString());
            chdDataset.addValue(row.getLong(2) - row.getLong(1), "Non TenYearCHD", row.get(0).toString());
        }

// Création du graphique
        JFreeChart chdChart = ChartFactory.createBarChart(
                "TenYearCHD distribution with BPMeds", "BPMeds", "Number of patients", chdDataset
        );

// Affichage du graphique dans une fenêtre Swing
        ChartPanel chdChartPanel = new ChartPanel(chdChart);
        JFrame chdFrame = new JFrame("TenYearCHD distribution with BPMeds");
        chdFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        chdFrame.add(chdChartPanel);
        chdFrame.pack();
        chdFrame.setVisible(true);

        //spark.stop();

    }

}
