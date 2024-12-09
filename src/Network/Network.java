/*
 * Author: Liam D. Tangney
 */

package Network;

import java.util.List;
import java.util.ArrayList;
import java.util.function.Function;
import java.util.concurrent.ThreadLocalRandom;

import Utils.ActivationFunctions;

public class Network {
    private Integer verbosity = null;
    private Double initialWeight = null;
    private Neuron biasNeuron = null;
    private List<Neuron> inputLayer = null;
    private List<List<Neuron>> hiddenLayers = null;
    private List<Neuron> outputLayer = null;
    private List<Edge> edges = null;
    private Function<Double, Double> activationFunction = null;
    private Function<Double, Double> activationFunctionPrime = null;

    public Network(
        Integer inputNeuronCount,
        Integer outputNeuronCount,
        List<Integer> hiddenLayerCounts,
        Double initialWeight,
        Integer verbosity,
        Function<Double, Double> activationFunction,
        Function<Double, Double> activationFunctionPrime
    ) {
        this.setInitialWeight(initialWeight);
        this.setVerbosity(verbosity);
        this.setActivationFunction(activationFunction, activationFunctionPrime);

        this.edges = new ArrayList<>();
        this.setInputLayerBySize(inputNeuronCount);
        this.setHiddenLayersBySize(hiddenLayerCounts);
        this.setOutputLayerBySize(outputNeuronCount);
        this.initializeBiasNeuron();
        this.addEdges();
    }

    public Double getInitialWeight() {
        return this.initialWeight;
    }

    public Double getRandomInitialWeight() {
        Double initialWeight = this.getInitialWeight();
        if (initialWeight == 0.0) {
            return 0.0;
        }

        return ThreadLocalRandom.current().nextDouble(0 - initialWeight, initialWeight);
    }

    public void setInitialWeight(Double weight) {
        this.initialWeight = weight;
    }

    public Integer getVerbosity() {
        return this.verbosity;
    }

    public void setVerbosity(Integer verbosity) {
        this.verbosity = verbosity;
    }

    public Neuron getBiasNeuron() {
        return this.biasNeuron;
    }

    public void setBiasNeuron(Neuron neuron) {
        this.biasNeuron = neuron;
    }

    private void initializeBiasNeuron() {
        Neuron biasNeuron = new Neuron();
        biasNeuron.setInput(1.0);
        this.setBiasNeuron(biasNeuron);
    }

    public List<Neuron> getInputLayer() {
        return this.inputLayer;
    }

    public void setInputLayer(List<Neuron> inputLayer) {
        this.inputLayer = new ArrayList<>(inputLayer);
    }

    public void setInputLayerBySize(Integer size) {
        this.inputLayer = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            this.inputLayer.add(new Neuron());
        }
    }

    public void setInputNeuron(Integer index, Neuron neuron) {
        this.inputLayer.set(index, neuron);
    }

    public List<List<Neuron>> getHiddenLayers() {
        return this.hiddenLayers;
    }

    public List<Neuron> getHiddenLayer(Integer index) {
        return this.hiddenLayers.get(index);
    }

    public void setHiddenLayers(List<List<Neuron>> hiddenLayers) {
        this.hiddenLayers = new ArrayList<>();
        for (List<Neuron> hiddenLayer : hiddenLayers) {
            this.hiddenLayers.add(new ArrayList<>(hiddenLayer));
        }
    }

    public void setHiddenLayersBySize(List<Integer> sizes) {
        List<Neuron> hiddenLayer = null;
        
        this.hiddenLayers = new ArrayList<>();
        for (Integer size : sizes) {
            hiddenLayer = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                hiddenLayer.add(
                    new Neuron(
                        this.getActivationFunction(),
                        this.getActivationFunctionPrime()
                    )
                );
            }

            this.hiddenLayers.add(hiddenLayer);
        }
    }

    public void setHiddenLayer(Integer index, List<Neuron> hiddenLayer) {
        this.hiddenLayers.set(index, new ArrayList<>(hiddenLayer));
    }

    public List<Neuron> getOutputLayer() {
        return this.outputLayer;
    }

    public void setOutputLayer(List<Neuron> outputLayer) {
        this.outputLayer = new ArrayList<>(outputLayer);
    }

    public void setOutputLayerBySize(Integer size) {
        this.outputLayer = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            this.outputLayer.add(
                new Neuron(
                    ActivationFunctions::logistic,
                    ActivationFunctions::logisticPrime,
                    NeuronType.Output
                )
            );
        }
    }

    public void setOutputNeuron(Integer index, Neuron neuron) {
        this.outputLayer.set(index, neuron);
    }

    /**
     * @return A list containing all hidden layers and the output layer.
     */
    public List<List<Neuron>> getLayers() {
        List<List<Neuron>> layers = new ArrayList<>();
        for (List<Neuron> hiddenLayer : this.getHiddenLayers()) {
            layers.add(hiddenLayer);
        }

        layers.add(this.getOutputLayer());
        return layers;
    }

    public List<Edge> getEdges() {
        return this.edges;
    }

    public void setEdges(List<Edge> edges) {
        this.edges = edges;
    }

    public void addEdge(Neuron source, Neuron destination) {
        Edge edge = new Edge(source, this.getRandomInitialWeight(), destination);
        source.addOutputEdge(edge);
        destination.addInputEdge(edge);
        this.edges.add(edge);
    }

    public Function<Double, Double> getActivationFunction() {
        return this.activationFunction;
    }

    public Function<Double, Double> getActivationFunctionPrime() {
        return this.activationFunctionPrime;
    }

    public void setActivationFunction(
        Function<Double, Double> activationFunction,
        Function<Double, Double> activationFunctionPrime
    ) {
        this.activationFunction = activationFunction;
        this.activationFunctionPrime = activationFunctionPrime;
    }

    /**
     * Adds weighted, directed edges from the bias neuron to each neuron in each hidden
     * layer and the output layer.
     * 
     * Edge weights are set to 1.
     */
    private void addBiasNeuronEdges() {
        for (List<Neuron> layer : this.getLayers()) {
            for (Neuron neuron : layer) {
                this.addEdge(this.getBiasNeuron(), neuron);
            }
        }
    }

    /**
     * Adds weighted, directed edges from every neuron in sourceLayer to every neuron in
     * destinationLayer.
     * 
     * @param sourceLayer Neuron layer consisting of nodes where edges begin.
     * @param destinationLayer Neuron layer consisting of nodes where edges are directed to.
     */
    private void connectLayers(List<Neuron> sourceLayer, List<Neuron> destinationLayer) {
        for (Neuron source : sourceLayer) {
            for (Neuron destination : destinationLayer) {
                this.addEdge(source, destination);
                // this.addEdge(this.getRandomInitialWeight(), source, destination);
            }
        }
    }

    /**
     * Adds weighted, directed edges from every node in the input layer to either the
     * first hidden layer or the output layer if there are no hidden layers.
     */
    private void addInputLayerEdges() {
        List<Neuron> nextLayer = null;

        if (this.getHiddenLayers().size() != 0) {
            nextLayer = this.getHiddenLayers().get(0);
        } else {
            nextLayer = this.getOutputLayer();
        }

        this.connectLayers(this.getInputLayer(), nextLayer);
    }

    /**
     * Adds weighted, directed edges between each hidden layer.
     * 
     * Adds weighted, directed edges between the final hidden layer and the output layer.
     * 
     * If no hidden layers exist, this method does nothing.
     */
    private void addHiddenLayerEdges() {
        if (this.getHiddenLayers().size() == 0) {
            return;
        }

        for (int i = 0; i < this.getHiddenLayers().size() - 1; i++) {
            this.connectLayers(this.getHiddenLayer(i), this.getHiddenLayer(i + 1));
        }

        this.connectLayers(
            this.getHiddenLayer(this.getHiddenLayers().size() - 1),
            this.getOutputLayer()
        );
    }

    /**
     * Adds weighted, directed edges between each layer.
     * 
     * This method does not handle edges originating from the bias neuron.
     */
    private void addEdges() {
        this.addBiasNeuronEdges();
        this.addInputLayerEdges();
        this.addHiddenLayerEdges();
    }

    /**
     * Feeds a data set to the network using forward propagaion.
     * 
     * @param data Input data set.
     */
    public void feed(List<Double> data) {
        for (int i = 0; i < this.getInputLayer().size(); i++) {
            this.getInputLayer().get(i).setInput(data.get(i));
        }

        for (List<Neuron> layer : this.getLayers()) {
            for (Neuron neuron : layer) {
                neuron.update();
            }
        }
    }

    public void backpropagate(
        List<Double> inputs,
        List<Integer> outputClass
    ) throws NetworkException {
        if (inputs.size() != this.getInputLayer().size()) {
            throw new NetworkException(
                String.format(
                    "Invalid input vector size: %d should be %d",
                    inputs.size(),
                    this.getInputLayer().size()
                )
            );
        }

        /* Forward Propagating */
        this.feed(inputs);

        /* Backward Propagating */
        for (int i = 0; i < this.getOutputLayer().size(); i++) {
            this.getOutputLayer().get(i).computeDelta(outputClass.get(i));
        }

        for (int i = this.getHiddenLayers().size() - 1; i >= 0; i--) {
            for (Neuron neuron : this.getHiddenLayer(i)) {
                neuron.computeDelta();
            }
        }

        /* Adding newly calculated values to the caches */
        for (Edge edge : this.getEdges()) {
            edge.updateCaches();
        }
    }

    private Integer getMaxValueIndex(List<Double> values) {
        Double maxValue = null;
        Integer maxValueIndex = null;

        if (values.size() == 0) {
            return null;
        }

        maxValue = values.get(0);
        maxValueIndex = 0;
        for (int i = 1; i < values.size(); i++) {
            if (values.get(i) > maxValue) {
                maxValue = values.get(i);
                maxValueIndex = i;
            }
        }

        return maxValueIndex;
    }

    public Integer getDecodedOutput() {
        List<Double> outputValues = new ArrayList<>();
        for (Neuron neuron : this.getOutputLayer()) {
            outputValues.add(neuron.getOutput());
        }

        return this.getMaxValueIndex(outputValues) + 1;
    }

    /**
     * @return The scaled, encoded output class.
     */
    public List<Double> getOutput() {
        List<Double> outputValues = new ArrayList<>();
        for (Neuron neuron : this.getOutputLayer()) {
            outputValues.add(neuron.getOutput());
        }

        return outputValues;
    }

    public Integer run(List<Double> data) throws NetworkException {
        this.feed(data);

        return this.getDecodedOutput();
    }
}
