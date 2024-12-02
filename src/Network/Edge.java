/*
 * Author: Liam D. Tangney
 */

package Network;

public class Edge {
    private Double weight = null;
    private Neuron source = null;
    private Neuron destination = null;

    public Edge(Neuron source, Double weight) {
        this.setSource(source);
        this.setWeight(weight);
    }

    public Edge(Neuron source, Double weight, Neuron destination) {
        this(source, weight);
        this.setDestination(destination);
    }

    public Double getWeight() {
        return this.weight;
    }

    public void setWeight(Double weight) {
        this.weight = weight;
    }

    public Neuron getSource() {
        return this.source;
    }

    public void setSource(Neuron source) {
        this.source = source;
    }

    public Neuron getDestination() {
        return this.destination;
    }

    public void setDestination(Neuron destination) {
        this.destination = destination;
    }
}
