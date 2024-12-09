/*
 * Author: Liam D. Tangney
 */

package Network;

import java.util.List;
import java.util.ArrayList;

public class Edge {
    private Double weight = null;
    private Neuron source = null;
    private Neuron destination = null;
    private List<Double> aICache = null;
    private List<Double> deltaJCache = null;

    public Edge(Neuron source, Double weight) {
        this.setSource(source);
        this.setWeight(weight);
        this.setAICache(new ArrayList<>());
        this.setDeltaJCache(new ArrayList<>());
    }

    public Edge(Neuron source, Double weight, Neuron destination) {
        this(source, weight);
        this.setDestination(destination);
    }

    public List<Double> getAICache() {
        return this.aICache;
    }

    public void setAICache(List<Double> aICache) {
        this.aICache = aICache;
    }

    public List<Double> getDeltaJCache() {
        return this.deltaJCache;
    }

    public void setDeltaJCache(List<Double> deltaJCache) {
        this.deltaJCache = deltaJCache;
    }

    public void clearCaches() {
        this.setAICache(new ArrayList<>());
        this.setDeltaJCache(new ArrayList<>());
    }

    public void updateCaches() {
        this.getAICache().add(this.getSource().getOutput());
        this.getDeltaJCache().add(this.getDestination().getDelta());
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

    @Override
    public boolean equals(Object o) {
        Edge edge = null;

        if (!(o instanceof Edge)) {
            return false;
        }

        edge = (Edge) o;
        return (
            this.getWeight() == edge.getWeight()
                && this.getSource().equals(edge.getSource())
                && this.getDestination().equals(edge.getDestination())
        );
    }
}
