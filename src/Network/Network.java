/*
 * Author: Liam D. Tangney
 */

package Network;

import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;

public class Network {
    private Integer verbosity = null;
    private Double initialWeight = null;
    private Neuron biasNeuron = null;
    private List<Neuron> inputLayer = null;
    private List<List<Neuron>> hiddenLayers = null;
    private List<Neuron> outputLayer = null;
    private List<Edge> edges = null;
    private Map<Neuron, List<Edge>> edgeToMap = null;
    private Map<Neuron, List<Edge>> edgeFromMap = null;

    public Network(
        Integer inputNeuronCount,
        Integer outputNeuronCount,
        List<Integer> hiddenLayerCounts,
        Double initialWeight,
        Integer verbosity
    ) throws NetworkException {
        this.edges = new ArrayList<>();
        this.edgeToMap = new HashMap<>();
        this.edgeFromMap = new HashMap<>();

        this.setInitialWeight(initialWeight);
        this.setVerbosity(verbosity);

        this.setInputLayerBySize(inputNeuronCount);
        this.setHiddenLayersBySize(hiddenLayerCounts);
        this.setOutputLayerBySize(outputNeuronCount);
        this.initializeBiasNeuron();
        this.addEdges();
    }

    private void addEdge(Double weight, Neuron source, Neuron destination) {
        List<Edge> edgeList = null;
        Edge edge = new Edge(source, weight, destination);
        this.edges.add(edge);

        if (this.edgeToMap.containsKey(destination)) {
            this.edgeToMap.get(destination).add(edge);
        } else {
            edgeList = new ArrayList<>();
            edgeList.add(edge);
            this.edgeToMap.put(destination, edgeList);
        }

        if (this.edgeFromMap.containsKey(source)) {
            this.edgeFromMap.get(source).add(edge);
        } else {
            edgeList = new ArrayList<>();
            edgeList.add(edge);
            this.edgeFromMap.put(source, edgeList);
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
                // source.addEdgeTo(destination, this.getRandomInitialWeight());
                this.addEdge(this.getRandomInitialWeight(), source, destination);
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
        this.addInputLayerEdges();
        this.addHiddenLayerEdges();
    }

    public Double getInitialWeight() {
        return this.initialWeight;
    }

    public Double getRandomInitialWeight() {
        Double initialWeight = this.getInitialWeight();
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

    /**
     * Adds weighted, directed edges from the bias neuron to each neuron in each hidden
     * layer and the output layer.
     * 
     * Edge weights are set to 1.
     */
    private void updateBiasNeuron() {
        for (List<Neuron> hiddenLayer : this.getHiddenLayers()) {
            for (Neuron neuron : hiddenLayer) {
                // this.getBiasNeuron().addEdgeTo(neuron, 1.0);
                this.addEdge(
                    this.getRandomInitialWeight(),
                    this.getBiasNeuron(),
                    neuron
                );
            }
        }

        for (Neuron neuron : this.getOutputLayer()) {
            // this.getBiasNeuron().addEdgeTo(neuron, 1.0);
            this.addEdge(this.getRandomInitialWeight(), this.getBiasNeuron(), neuron);
        }
    }

    public void setBiasNeuron(Neuron neuron) {
        this.biasNeuron = neuron;
        this.updateBiasNeuron();
    }

    private void initializeBiasNeuron() throws NetworkException {
        Neuron biasNeuron = new Neuron(NeuronType.Input);
        biasNeuron.setInput(1.0);
        biasNeuron.setOutput(1.0);
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
            this.inputLayer.add(new Neuron(NeuronType.Input));
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
                hiddenLayer.add(new Neuron());
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
            this.outputLayer.add(new Neuron(NeuronType.Output));
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

    /**
     * Feeds a data set to the network using forward propagaion.
     * 
     * @param data Input data set.
     * @throws NetworkException Neuron is activated before getting data.
     */
    public void feed(List<Double> data) throws NetworkException {
        List<Double> currentValues = null;
        List<Edge> edges = null;

        for (int i = 0; i < this.getInputLayer().size(); i++) {
            this.getInputLayer().get(i).setInput(data.get(i));
        }

        for (List<Neuron> layer : this.getLayers()) {
            for (Neuron neuron : layer) {
                currentValues = new ArrayList<>();
                edges = this.edgeToMap.get(neuron);
                for (Edge edge : edges) {
                    currentValues.add(edge.getSource().getOutput());
                }

                neuron.setInput(currentValues, edges);
            }
        }
    }

    public void backpropagate(List<Double> inputs, Integer target) throws NetworkException {
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
            this.getOutputLayer().get(i).computeDelta(target);
        }

        for (int i = this.getHiddenLayers().size() - 1; i >= 0; i--) {
            for (Neuron neuron : this.getHiddenLayer(i)) {
                neuron.computeDelta(this.edgeFromMap.get(neuron));
            }
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

    public Integer getOutput() {
        List<Double> outputValues = new ArrayList<>();
        for (Neuron neuron : this.getOutputLayer()) {
            outputValues.add(neuron.getOutput());
        }

        return this.getMaxValueIndex(outputValues) + 1;
    }

    public Integer run(List<Double> data) throws NetworkException {
        this.feed(data);

        return this.getOutput();
    }
}
