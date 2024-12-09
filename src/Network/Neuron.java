/*
 * Author: Liam D. Tangney
 */

package Network;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class Neuron {
    private Double input = null;
    private Double output = null;
    private Double delta = null;
    private NeuronType type = null;
    private Function<Double, Double> activationFunction = null;
    private Function<Double, Double> activationFunctionPrime = null;
    private List<Edge> inputEdges = null;
    private List<Edge> outputEdges = null;

    public Neuron(
        Function<Double, Double> activationFunction,
        Function<Double, Double> activationFunctionPrime,
        NeuronType type
    ) {
        this.setType(type);
        this.setActivationFunction(activationFunction, activationFunctionPrime);
        this.inputEdges = new ArrayList<>();
        this.outputEdges = new ArrayList<>();
    }

    public Neuron(
        Function<Double, Double> activationFunction,
        Function<Double, Double> activationFunctionPrime
    ) {
        this(activationFunction, activationFunctionPrime, NeuronType.Hidden);
    }

    public Neuron() {
        this((input) -> {return input;}, null, NeuronType.Input);
    }

    /**
     * Removes all input edges from neuron.
     * 
     * @param neuron The neuron in which connecting edges will be removed.
     */
    public void removeEdgesFrom(Neuron neuron) {
        List<Edge> duplicates = new ArrayList<>();
        for (Edge edge : this.getInputEdges()) {
            if (edge.getSource().equals(neuron)) {
                duplicates.add(edge);
            }
        }

        for (Edge edge : duplicates) {
            this.getInputEdges().remove(edge);
        }
    }

    /**
     * Removes all output edges to neuron.
     * 
     * @param neuron The neuron in which connecting edges will be removed.
     */
    public void removeEdgesTo(Neuron neuron) {
        List<Edge> duplicates = new ArrayList<>();
        for (Edge edge : this.getOutputEdges()) {
            if (edge.getDestination().equals(neuron)) {
                duplicates.add(edge);
            }
        }

        for (Edge edge : duplicates) {
            this.getOutputEdges().remove(edge);
        }
    }

    public List<Edge> getInputEdges() {
        return this.inputEdges;
    }

    public void setInputEdges(List<Edge> edges) {
        this.inputEdges = edges;
    }

    public void addInputEdge(Edge edge) {
        if (!edge.getDestination().equals(this)) {
            return;
        }

        this.removeEdgesFrom(edge.getSource());
        this.inputEdges.add(edge);
    }

    public void addInputEdge(Neuron source, Double weight) {
        this.addInputEdge(new Edge(source, weight, this));
    }

    public List<Edge> getOutputEdges() {
        return this.outputEdges;
    }

    public void setOutputEdges(List<Edge> edges) {
        this.outputEdges = edges;
    }

    public void addOutputEdge(Edge edge) {
        if (!edge.getSource().equals(this)) {
            return;
        }

        this.removeEdgesTo(edge.getDestination());
        this.outputEdges.add(edge);
    }

    public void addOutputEdge(Neuron destination, Double weight) {
        this.addOutputEdge(new Edge(this, weight, destination));
    }

    private void activate() {
        this.setOutput(this.getActivationFunction().apply(this.getInput()));
    }

    public void update() {
        Double sum = 0.0;
        for (Edge edge : this.getInputEdges()) {
            sum += edge.getSource().getOutput() * edge.getWeight();
        }

        this.setInput(sum);
        this.activate();
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

    public NeuronType getType() {
        return this.type;
    }

    public void setType(NeuronType type) {
        this.type = type;
    }

    public Double getInput() {
        return this.input;
    }

    public void setInput(Double input) {
        this.input = input;
        this.activate();
    }

    public Double getOutput() {
        return this.output;
    }

    private void setOutput(Double output) {
        this.output = output;
    }

    public Double getDelta() {
        return this.delta;
    }

    public void setDelta(Double delta) {
        this.delta = delta;
    }

    public Double getPrimeActivation() {
        return this.getActivationFunctionPrime().apply(this.getInput());
    }

    public void computeDelta(Integer y) throws NetworkException {
        if (this.getOutput() == null || this.getInput() == null) {
            throw new NetworkException("Cannot calculate delta without input or output");
        }

        if (this.type != NeuronType.Output) {
            throw new NetworkException("Cannot use this formula on a non-output neuron");
        }

        this.setDelta(this.getPrimeActivation() * (-2.0 * (y - this.getOutput())));
    }

    public void computeDelta() throws NetworkException {
        if (this.getOutput() == null || this.getInput() == null) {
            throw new NetworkException("Cannot calculate delta without input or output");
        }

        if (this.type != NeuronType.Hidden) {
            throw new NetworkException("Cannot use this formula on a non-hidden neuron");
        }

        Double sum = 0.0;
        for (Edge edge : this.getOutputEdges()) {
            sum += edge.getWeight() * edge.getDestination().getDelta();
        }

        this.setDelta(this.getPrimeActivation() * sum);
    }
}
