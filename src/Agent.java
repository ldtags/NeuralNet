/*
 * Author: Liam D. Tangney
 */

import java.util.List;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.FileReader;
import java.util.ArrayList;

public class Agent {
    private List<Integer> hiddenLayers = null;
    private Double learningRate = 0.01;
    private Integer epochLimit = 1000;
    private Integer batchSize = 1;
    private Double regularization = 0.0;
    private Boolean randomized = false;
    private Double weightInit = 0.1;
    private Integer verbosity = 1;
    private List<DataPoint> data = null;

    public Agent() {
        this.hiddenLayers = new ArrayList<>();
        this.data = new ArrayList<>();
    }

    public List<Integer> getHiddenLayers() {
        return this.hiddenLayers;
    }

    public void setHiddenLayers(List<Integer> hiddenLayers) {
        this.hiddenLayers = new ArrayList<>(hiddenLayers);
    }

    public Double getLearningRate() {
        return this.learningRate;
    }

    public void setLearningRate(Double learningRate) {
        this.learningRate = learningRate;
    }

    public Integer getEpochLimit() {
        return this.epochLimit;
    }

    public void setEpochLimit(Integer epochLimit) {
        this.epochLimit = epochLimit;
    }

    public Integer getBatchSize() {
        return this.batchSize;
    }

    public void setBatchSize(Integer batchSize) {
        this.batchSize = batchSize;
    }

    public Double getRegularization() {
        return this.regularization;
    }

    public void setRegularization(Double regularization) {
        this.regularization = regularization;
    }

    public Boolean getRandomization() {
        return this.randomized;
    }

    public void setRandomization(Boolean randomized) {
        this.randomized = randomized;
    }

    public Double getWeightInitialization() {
        return this.weightInit;
    }

    public void setWeightInitialization(Double weight) {
        this.weightInit = weight;
    }

    public Integer getVerbosity() {
        return this.verbosity;
    }

    public void setVerbosity(Integer verbosity) {
        if (verbosity < 1 || verbosity > 4) {
            throw new NumberFormatException("Verbosity must be either [1 | 2 | 3 | 4]");
        }

        this.verbosity = verbosity;
    }

    public List<DataPoint> getData() {
        return this.data;
    }

    public void setData(List<DataPoint> data) {
        this.data = new ArrayList<>(data);
    }

    private static String cleanRawData(String rawData) {
        return rawData
            .replace("(", "")
            .replace(")", "")
            .strip();
    }

    public void loadData(String filename) throws FileNotFoundException, IOException, NumberFormatException {
        String line = null, featureString = null, targetString = null;
        Integer index = null;
        BufferedReader reader = null;
        List<Double> features = null;
        List<Integer> targets = null;
        List<DataPoint> data = new ArrayList<>();
    
        try{
            reader = new BufferedReader(new FileReader(filename));
            while ((line = reader.readLine()) != null) {
                line = line.strip();
                if (line.length() == 0 || line.charAt(0) == '#') {
                    continue;
                }

                for (int i = 1; i < line.length(); i++) {
                    if (line.charAt(i) == '(') {
                        index = i - 1;
                        break;
                    }
                }

                if (index == null) {
                    continue;
                }

                featureString = cleanRawData(line.substring(0, index));
                targetString = cleanRawData(line.substring(index));

                features = new ArrayList<>();
                for (String feature : featureString.split(" ")) {
                    features.add(Double.parseDouble(feature));
                }

                targets = new ArrayList<>();
                for (String target : targetString.split(" ")) {
                    targets.add(Integer.parseInt(target));
                }

                data.add(new DataPoint(features, targets));
            }

            this.setData(data);
        } finally {
            reader.close();
        }
    }
}
