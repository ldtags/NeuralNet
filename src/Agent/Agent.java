/*
 * Author: Liam D. Tangney
 */

package Agent;

import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;
import java.util.HashMap;

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

    /**
     * Splits data points into training and validation sets.
     * 
     * @param data : List of data points being split.
     * @param randomize : Specifies if the order of the data set should be randomized.
     * @return Mapping of training and validation sets to the respective keywords
     *      "training" and "validation".
     */
    private static Map<String, List<DataPoint>> splitDataPoints(List<DataPoint> data, boolean randomize) {
        Integer trainingSetSize = null;
        List<DataPoint> inputData = null;
        Map<String, List<DataPoint>> splitData = null;

        inputData = new ArrayList<>(data);
        if (randomize) {
            Collections.shuffle(inputData);
        }

        splitData = new HashMap<>();
        trainingSetSize = Math.ceilDiv(inputData.size() * 4, 5);
        splitData.put("training", inputData.subList(0, trainingSetSize));
        splitData.put("validation", inputData.subList(trainingSetSize, inputData.size()));
        return splitData;
    }

    private static List<Double> getFeatureMinimums(List<DataPoint> data) {
        Double feature = null;
        List<Double> features = null, featureMinimums = null;

        if (data.size() == 0) {
            return new ArrayList<>();
        }

        featureMinimums = data.get(0).getFeatures();
        for (int i = 1; i < data.size(); i++) {
            features = data.get(i).getFeatures();
            for (int j = 0; j < features.size(); j++) {
                feature = features.get(j);
                if (j > featureMinimums.size()) {
                    featureMinimums.add(features.get(j));
                } else if (feature < featureMinimums.get(j)) {
                    featureMinimums.set(j, feature);
                }
            }
        }

        return featureMinimums;
    }

    private static List<Double> getFeatureMaximums(List<DataPoint> data) {
        Double feature = null;
        List<Double> features = null, featureMaximums = null;

        if (data.size() == 0) {
            return new ArrayList<>();
        }

        featureMaximums = data.get(0).getFeatures();
        for (int i = 1; i < data.size(); i++) {
            features = data.get(i).getFeatures();
            for (int j = 0; j < features.size(); j++) {
                feature = features.get(j);
                if (j > featureMaximums.size()) {
                    featureMaximums.add(features.get(j));
                } else if (feature > featureMaximums.get(j)) {
                    featureMaximums.set(j, feature);
                }
            }
        }

        return featureMaximums;
    }

    private static void scaleDataPoints(List<DataPoint> data, List<Double> featureMinimums, List<Double> featureMaximums) {
        Double min = null, max = null, scalar = null;
        List<Double> features = null;

        for (DataPoint dataPoint : data) {
            features = dataPoint.getFeatures();
            for (int i = 0; i < features.size(); i++) {
                min = featureMinimums.get(i);
                max = featureMaximums.get(i);
                if (min == max) {
                    scalar = 0.0;
                } else {
                    scalar = (features.get(i) - min) / (max - min);
                }

                dataPoint.setFeature(i, -1 + 2 * scalar);
            }
        }
    }

    public void start() throws AgentException {
        Map<String, List<DataPoint>> splitData = null;
        List<Double> featureMinimums = null, featureMaximums = null;
        List<DataPoint> trainingSet = null, validationSet = null;

        if (this.getData().size() == 0) {
            throw new AgentException("No data found, please load data into agent");
        }

        splitData = splitDataPoints(this.getData(), this.getRandomization());
        if (!splitData.containsKey("training") || !splitData.containsKey("validation")) {
            throw new AgentException("An error occurred while splitting input data");
        }

        trainingSet = splitData.get("training");
        validationSet = splitData.get("validation");
        featureMinimums = getFeatureMinimums(trainingSet);
        featureMaximums = getFeatureMaximums(trainingSet);
        scaleDataPoints(trainingSet, featureMinimums, featureMaximums);
        scaleDataPoints(validationSet, featureMinimums, featureMaximums);
    }
}
