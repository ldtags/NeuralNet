/*
 * Author: Liam D. Tangney
 */

package Network;

import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;

public class Network {
    private Double initialWeight = null;
    private Neuron biasNeuron = null;
    private List<Neuron> inputLayer = null;
    private List<List<Neuron>> hiddenLayers = null;
    private List<Neuron> outputLayer = null;
 
    public Network(Integer inputNeuronCount, Integer outputNeuronCount, List<Integer> hiddenLayerCounts, Double initialWeight) {
        this.setInitialWeight(initialWeight);
        this.setInputLayerBySize(inputNeuronCount);
        this.setHiddenLayersBySize(hiddenLayerCounts);
        this.setOutputLayerBySize(outputNeuronCount);
        this.initializeBiasNeuron();
        this.addEdges();
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
                source.addEdgeTo(destination, this.getRandomInitialWeight());
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
                this.getBiasNeuron().addEdgeTo(neuron, 1.0);
            }
        }

        for (Neuron neuron : this.getOutputLayer()) {
            this.getBiasNeuron().addEdgeTo(neuron, 1.0);
        }
    }

    public void setBiasNeuron(Neuron neuron) {
        this.biasNeuron = neuron;
        this.updateBiasNeuron();
    }

    private void initializeBiasNeuron() {
        this.setBiasNeuron(new Neuron());
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
            this.outputLayer.add(new Neuron());
        }
    }

    public void setOutputNeuron(Integer index, Neuron neuron) {
        this.outputLayer.set(index, neuron);
    }
}
