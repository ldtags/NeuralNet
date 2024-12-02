/*
 * Author: Liam D. Tangney
 */

package Network;

import java.util.List;
import java.util.ArrayList;

public class Neuron {
    private List<Edge> edges = null;

    public Neuron() {
        this.edges = new ArrayList<>();
    }

    public List<Edge> getEdges() {
        return this.edges;
    }

    public Edge getEdge(Integer index) {
        return this.getEdges().get(index);
    }

    public void setEdges(List<Edge> edges) {
        this.edges = new ArrayList<>(edges);
    }

    public void setEdge(Integer index, Edge edge) {
        this.edges.set(index, edge);
    }

    public void removeEdgesTo(Neuron neuron) {
        List<Edge> duplicateEdges = null;

        if (neuron == null) {
            return;
        }

        duplicateEdges = new ArrayList<>();
        for (Edge edge : this.getEdges()) {
            if (edge.getDestination().equals(neuron)) {
                duplicateEdges.add(edge);
            }
        }

        for (Edge edge : duplicateEdges) {
            this.getEdges().remove(edge);
        }
    }

    public void addEdge(Edge edge) {
        this.removeEdgesTo(edge.getDestination());
        this.edges.add(edge);
    }

    public void addEdgeTo(Neuron neuron, Double weight) {
        this.removeEdgesTo(neuron);
        this.edges.add(new Edge(this, weight, neuron));
    }

    public Double activate(Double input) {
        return 1.0 / (1.0 + Math.exp(input));
    }

    @Override
    public boolean equals(Object o) {
        Neuron neuron = null;

        if (!(o instanceof Neuron)) {
            return false;
        }

        neuron = (Neuron) o;
        if (this.getEdges().size() != neuron.getEdges().size()) {
            return false;
        }

        for (int i = 0; i < this.getEdges().size(); i++) {
            if (!this.getEdge(i).equals(neuron.getEdge(i))) {
                return false;
            }
        }

        return true;
    }
}
