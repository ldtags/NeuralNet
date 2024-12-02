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

    public void addEdge(Edge edge) {
        this.edges.add(edge);
    }
}
