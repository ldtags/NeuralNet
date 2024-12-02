/*
 * Author: Liam D. Tangney
 */

package Network;

import java.util.List;
import java.util.ArrayList;

public class Network {
    private List<Neuron> inputLayer = null;
    private List<List<Neuron>> hiddenLayers = null;
    private List<Neuron> outputLayer = null;
 
    public Network(Integer inputNeuronCount, Integer outputNeuronCount, List<Integer> hiddenLayerCounts, Double initialWeight) {
        Neuron neuron = null;
        List<Neuron> hiddenLayer = null;

        this.inputLayer = new ArrayList<>();
        for (int i = 0; i < inputNeuronCount; i++) {
            neuron = new Neuron();
            neuron.addEdge(new Edge(neuron, initialWeight));
            this.inputLayer.add(neuron);
        }

        this.hiddenLayers = new ArrayList<>();
        for (Integer hiddenLayerCount : hiddenLayerCounts) {
            hiddenLayer = new ArrayList<>();
            for (int i = 0; i < hiddenLayerCount; i++) {
                neuron = new Neuron();
                neuron.addEdge(new Edge(neuron, initialWeight));
                hiddenLayer.add(neuron);
            }

            this.hiddenLayers.add(hiddenLayer);
        }

        this.outputLayer = new ArrayList<>();
        for (int i = 0; i < outputNeuronCount; i++) {
            neuron = new Neuron();
            neuron.addEdge(new Edge(neuron, initialWeight));
            this.outputLayer.add(neuron);
        }
    }

    public List<Neuron> getInputLayer() {
        return this.inputLayer;
    }

    public void setInputLayer(List<Neuron> inputLayer) {
        this.inputLayer = new ArrayList<>(inputLayer);
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

    public void setHiddenLayer(Integer index, List<Neuron> hiddenLayer) {
        this.hiddenLayers.set(index, new ArrayList<>(hiddenLayer));
    }

    public List<Neuron> getOutputLayer() {
        return this.outputLayer;
    }

    public void setOutputLayer(List<Neuron> outputLayer) {
        this.outputLayer = new ArrayList<>(outputLayer);
    }

    public void setOutputNeuron(Integer index, Neuron neuron) {
        this.outputLayer.set(index, neuron);
    }
}
