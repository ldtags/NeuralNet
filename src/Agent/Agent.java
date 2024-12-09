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

import Models.DataPoint;
import Network.Edge;
import Network.Neuron;
import Network.Network;
import Network.NetworkException;

public class Agent {
    private List<Integer> hiddenLayerSizes = null;
    private Double learningRate = 0.01;
    private Integer epochLimit = 1000;
    private Integer batchSize = 1;
    private Double regularization = 0.0;
    private Boolean randomized = false;
    private Double weightInit = 0.1;
    private Integer verbosity = 1;
    private List<DataPoint> data = null;

    public Agent() {
        this.hiddenLayerSizes = new ArrayList<>();
        this.data = new ArrayList<>();
    }

    public List<Integer> getHiddenLayerSizes() {
        return this.hiddenLayerSizes;
    }

    public Integer getHiddenLayerSize(Integer index) {
        return this.getHiddenLayerSizes().get(index);
    }

    public void setHiddenLayerSizes(List<Integer> hiddenLayers) {
        this.hiddenLayerSizes = new ArrayList<>(hiddenLayers);
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

    public void setWeightInitialization(Double weight) throws AgentException {
        if (weight < 0) {
            throw new AgentException(
                "Weight initializer must be a non-negative integer"
            );
        }

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

    public Integer getNumberOfFeatures() {
        List<DataPoint> dataSet = this.getData();
        Integer featureMax = null, size = null;

        if (dataSet.size() == 0) {
            return 0;
        }

        featureMax = dataSet.get(0).getFeatures().size();
        for (int i = 1; i < dataSet.size(); i++) {
            size = dataSet.get(i).getFeatures().size();
            if (size > featureMax) {
                featureMax = size;
            }
        }

        return featureMax;
    }

    public Integer getNumberOfClasses() {
        List<DataPoint> dataSet = this.getData();
        Integer classMax = null, size = null;

        if (dataSet.size() == 0) {
            return 0;
        }

        classMax = dataSet.get(0).getOutputClass().size();
        for (int i = 1; i < dataSet.size(); i++) {
            size = dataSet.get(i).getOutputClass().size();
            if (size > classMax) {
                classMax = size;
            }
        }

        return classMax;
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
    
        System.out.printf("* Reading %s\n", filename);
        try {
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
        trainingSetSize = (int) Math.ceil((inputData.size() * 4.0) / 5.0);
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

        featureMinimums = new ArrayList<>(data.get(0).getFeatures());
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

        featureMaximums = new ArrayList<>(data.get(0).getFeatures());
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

    private static void scaleDataSet(List<DataPoint> data, List<Double> featureMinimums, List<Double> featureMaximums) {
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

    /**
     * Splits data into a collection of batches. Each batch will be as evenly sized
     * as possible.
     * 
     * @param data Data set being split into batches
     * 
     * @return The collection of batches
     */
    private List<List<DataPoint>> getBatches(List<DataPoint> data) {
        int batchCount, batchNum;
        List<DataPoint> randomData = null, batch = null;
        List<List<DataPoint>> batches = null;

        switch (this.getBatchSize()) {
        // Full-Batch Gradient Descent
        case 0:
            batches = new ArrayList<>(1);
            batches.add(data);
            break;

        // Stochastic Gradient Descent
        case 1:
            batches = new ArrayList<>();
            for (DataPoint dataPoint : data) {
                batch = new ArrayList<>();
                batch.add(dataPoint);
                batches.add(batch);
            }

            break;

        // Mini-Batch Gradient Descent
        default:
            batchCount = (int) Math.ceil(
                (1.0 * data.size()) / (1.0 * this.getBatchSize())
            );
            batches = new ArrayList<>(batchCount);
            if (this.randomized) {
                randomData = new ArrayList<>(data);
                Collections.shuffle(randomData);
                data = randomData;
            }

            for (int i = 0; i < data.size(); i++) {
                batchNum = Math.floorDiv(i, this.getBatchSize());
                if (batches.size() <= batchNum) {
                    batches.add(new ArrayList<>());
                }

                batches.get(batchNum).add(data.get(i));
            }

            break;
        }

        return batches;
    }

    public static Double calculateLoss(Network network, DataPoint dataPoint) {
        Double sum = 0.0, actualOutput = null;
        List<Integer> outputClass = dataPoint.getOutputClass();

        for (int i = 0; i < outputClass.size(); i++) {
            actualOutput = network.getOutputLayer().get(i).getOutput();
            sum += Math.pow(outputClass.get(i) - actualOutput, 2);
        }

        return sum;
    }

    private Double calculateCost(Network network, List<DataPoint> data) {
        Double sum = 0.0, regFactor = null;

        if (data.size() == 0) {
            return sum;
        }

        for (DataPoint dataPoint : data) {
            sum += calculateLoss(network, dataPoint);
        }

        regFactor = 0.0;
        for (Edge edge : network.getEdges()) {
            regFactor += Math.pow(edge.getWeight(), 2);
        }

        regFactor *= this.getRegularization();
        return sum / (data.size() * 1.0) + regFactor;
    }

    private static Double calculateAccuracy(
        Network network,
        List<DataPoint> data
    ) throws NetworkException {
        Integer totalCorrect = 0;
        for (DataPoint dataPoint : data) {
            network.feed(dataPoint.getFeatures());
            if (dataPoint.getDecodedOutputClass() == network.getOutput()) {
                totalCorrect++;
            }
        }

        return (1.0 * totalCorrect) / (1.0 * data.size());
    }

    private void reportPreTrainingInfo(
        Network network,
        List<DataPoint> data
    ) throws NetworkException {
        Double lossSum = null;

        if (this.getVerbosity() >= 2) {
            switch (this.getBatchSize()) {
            case 0:
                System.out.println("  * Beginning full-batch gradient descent");
                break;
            case 1:
                System.out.println("  * Beginning stochastic gradient descent");
                break;
            default:
                System.out.println("  * Beginning mini-batch gradient descent");
                break;
            }

            System.out.printf(
                "    (batchSize=%d, epochLimit=%d, learningRate=%.4f, lambda=%.4f)\n",
                this.getBatchSize(),
                this.getEpochLimit(),
                this.getLearningRate(),
                this.getRegularization()
            );
        }

        if (this.getVerbosity() >= 3) {
            for (DataPoint dataPoint : data) {
                network.feed(dataPoint.getFeatures());
            }

            lossSum = 0.0;
            for (DataPoint dataPoint : data) {
                lossSum += calculateLoss(network, dataPoint);
            }

            System.out.printf(
                "    Initial model with random weights : Cost = %.6f; Loss = %.6f; Acc = %.4f\n",
                this.calculateCost(network, data),
                lossSum / (1.0 * data.size()),
                calculateAccuracy(network, data)
            );
        }
    }

    private void reportPostTrainingInfo(
        Long timeElapsed,
        Integer epochs,
        Integer iterations,
        String stopCondition
    ) {
        if (this.getVerbosity() >= 2) {
            System.out.println("  * Done with fitting!");
            System.out.printf(
                "    Training took %dms, %d epochs, %d iterations (%.4fms / iteration)\n",
                timeElapsed,
                epochs,
                iterations,
                (1.0 * timeElapsed) / (1.0 * iterations)
            );
            System.out.printf("    GD Stop condition: %s\n", stopCondition);
        }
    }

    private void reportEpochTrainingInfo(
        Network network,
        Integer epochs,
        Integer iterations,
        List<DataPoint> data
    ) throws NetworkException {
        Double lossSum = null;

        if (this.getVerbosity() >= 3) {
            lossSum = 0.0;
            for (DataPoint dataPoint : data) {
                lossSum += calculateLoss(network, dataPoint);
            }

            System.out.printf(
                "    After %6d epochs (%6d iter.): Cost = %.6f; Loss = %.6f; Acc = %.4f\n",
                epochs,
                iterations,
                this.calculateCost(network, data),
                lossSum / (1.0 * data.size()),
                calculateAccuracy(network, data)
            );
        }
    }

    private void reportNetworkState(
        Network network,
        Integer exampleNumber,
        List<Integer> actual
    ) {
        if (this.getVerbosity() >= 4) {
            System.out.printf(
                "    * Forward Propagation on example %d\n",
                exampleNumber
            );

            System.out.printf("      Layer 1 (input) : %7s", "a_j: ");
            System.out.printf("%2.3f ", 1.0);
            for (Neuron neuron : network.getInputLayer()) {
                System.out.printf("%2.3f ", neuron.getOutput());
            }
            System.out.println();

            for (int i = 0; i < network.getHiddenLayers().size(); i++) {
                System.out.printf("      Layer %d (hidden): %7s", i + 2, "in_j: ");
                for (Neuron neuron : network.getHiddenLayer(i)) {
                    System.out.printf("%2.3f ", neuron.getInput());
                }
                System.out.println();

                System.out.printf("                        %7s", "a_j: ");
                for (Neuron neuron : network.getHiddenLayer(i)) {
                    System.out.printf("%2.3f ", neuron.getOutput());
                }
                System.out.println();
            }

            System.out.printf(
                "      Layer %d (output): %7s",
                network.getHiddenLayers().size() + 2,
                "in_j: "
            );
            for (Neuron neuron : network.getOutputLayer()) {
                System.out.printf("%2.3f ", neuron.getInput());
            }
            System.out.println();

            System.out.printf("                        %7s", "a_j: ");
            for (Neuron neuron : network.getOutputLayer()) {
                System.out.printf("%2.3f ", neuron.getOutput());
            }
            System.out.println();

            System.out.print("           example's actual y: ");
            for (Integer value : actual) {
                System.out.printf("%2.3f ", value * 1.0);
            }
            System.out.println();

            System.out.printf(
                "    * Backward Propagation on example %d\n",
                exampleNumber
            );

            System.out.printf(
                "      Layer %d (output): %7s",
                network.getHiddenLayers().size() + 2,
                "Delta_j: "
            );
            for (Neuron neuron : network.getOutputLayer()) {
                System.out.printf("%2.3f ", neuron.getDelta());
            }
            System.out.println();

            for (int i = network.getHiddenLayers().size() - 1; i >= 0; i--) {
                System.out.printf("      Layer %d (hidden): %7s", i + 2, "Delta_j: ");
                for (Neuron neuron : network.getHiddenLayer(i)) {
                    System.out.printf("%2.3f ", neuron.getDelta());
                }
                System.out.println();
            }
            System.out.println();
        }
    }

    private static Double calculateDotProduct(List<Double> v1, List<Double> v2) {
        Double sum = null;

        if (v1.size() != v2.size()) {
            return null;
        }

        sum = 0.0;
        for (int i = 0; i < v1.size(); i++) {
            sum += v1.get(i) * v2.get(i);
        }

        return sum;
    }

    private void trainNetwork(
        Network network,
        List<DataPoint> trainingSet
    ) throws NetworkException {
        Integer t = 0, epochs = 0, exampleNumber = 1;
        Long startTime = null;
        String stopCondition = "Epoch Limit";
        List<List<DataPoint>> batches = null;

        this.reportPreTrainingInfo(network, trainingSet);
        startTime = System.currentTimeMillis();
        while (epochs < this.getEpochLimit()) {
            batches = this.getBatches(trainingSet);
            for (List<DataPoint> batch : batches) {
                /* Initialize empty caches */
                for (Edge edge : network.getEdges()) {
                    edge.clearCaches();
                }

                /* Calculating delta_j and a_i values via backpropagation */
                for (DataPoint dataPoint : batch) {
                    network.backpropagate(
                        dataPoint.getFeatures(),
                        dataPoint.getOutputClass()
                    );

                    this.reportNetworkState(
                        network,
                        exampleNumber++,
                        dataPoint.getOutputClass()
                    );
                }

                /* Updating edge weights via gradient descent */
                for (Edge edge : network.getEdges()) {
                    edge.setWeight(
                        edge.getWeight()
                        - (
                            this.getLearningRate()
                            * (
                                (1.0 / 1.0 * batch.size())
                                * calculateDotProduct(
                                    edge.getDeltaJCache(),
                                    edge.getAICache()
                                )
                            )
                        )
                        - (
                            2
                            * this.getLearningRate()
                            * this.getRegularization()
                            * edge.getWeight()
                        )
                    );
                }

                t++;
            }

            epochs++;
            if (this.getEpochLimit() < 10
                    || epochs % ((1.0 * this.getEpochLimit()) / 10.0) == 0) {
                this.reportEpochTrainingInfo(network, epochs, t, trainingSet);
            }
        }

        this.reportPostTrainingInfo(
            System.currentTimeMillis() - startTime,
            epochs,
            t,
            stopCondition
        );
    }

    private void reportNetworkInfo() {
        if (this.getVerbosity() > 1) {
            System.out.println("  * Layer sizes (excluding bias neuron(s)):");
            System.out.printf(
                "    Layer  1 (input) : %3d\n",
                this.getNumberOfFeatures()
            );
            for (int i = 0; i < this.getHiddenLayerSizes().size(); i++) {
                System.out.printf(
                    "    Layer %2d (hidden): %3d\n",
                    i + 2,
                    this.getHiddenLayerSize(i)
                );
            }
            System.out.printf(
                "    Layer %2d (output): %3d\n",
                this.getHiddenLayerSizes().size() + 2,
                this.getNumberOfClasses()
            );
        }
    }

    public void reportFeatureInfo(
        List<Double> featureMinimums,
        List<Double> featureMaximums
    ) {
        if (this.getVerbosity() > 1) {
            System.out.println("  * min/max values on training set:");
            for (int i = 0; i < this.getNumberOfFeatures(); i++) {
                System.out.printf(
                    "    Feature %d: %.3f, %.3f\n",
                    i + 1,
                    featureMinimums.get(i),
                    featureMaximums.get(i)
                );
            }
        }
    }

    public void start() throws AgentException {
        Network network = null;
        Map<String, List<DataPoint>> splitData = null;
        List<Double> featureMinimums = null, featureMaximums = null;
        List<DataPoint> trainingSet = null, validationSet = null;

        if (this.getData().size() == 0) {
            throw new AgentException("No data found, please load data into agent");
        }

        System.out.println("* Doing train/validation split");
        splitData = splitDataPoints(this.getData(), this.getRandomization());
        if (!splitData.containsKey("training") || !splitData.containsKey("validation")) {
            throw new AgentException("An error occurred while splitting input data");
        }

        trainingSet = splitData.get("training");
        validationSet = splitData.get("validation");

        System.out.println("* Scaling features");
        featureMinimums = getFeatureMinimums(trainingSet);
        featureMaximums = getFeatureMaximums(trainingSet);
        this.reportFeatureInfo(featureMinimums, featureMaximums);

        scaleDataSet(trainingSet, featureMinimums, featureMaximums);
        scaleDataSet(validationSet, featureMinimums, featureMaximums);

        try {
            System.out.println("* Building network");
            this.reportNetworkInfo();
            network = new Network(
                this.getNumberOfFeatures(),
                this.getNumberOfClasses(),
                this.getHiddenLayerSizes(),
                this.getWeightInitialization(),
                this.getVerbosity()
            );

            System.out.printf("* Training network (using %d examples)\n", trainingSet.size());
            this.trainNetwork(network, trainingSet);

            System.out.println("* Evaluating accuracy");
            System.out.printf(
                "  TrainAcc: %.6f\n",
                calculateAccuracy(network, trainingSet)
            );
            System.out.printf(
                "  ValidAcc: %.6f\n",
                calculateAccuracy(network, validationSet)
            );
        } catch (NetworkException e) {
            throw new AgentException(e.getMessage());
        }
    }
}
