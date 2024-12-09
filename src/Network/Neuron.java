/*
 * Author: Liam D. Tangney
 */

package Network;

import java.util.List;
import java.util.function.Function;

public class Neuron {
    private Double input = null;
    private Double output = null;
    private Double delta = null;
    private NeuronType type = null;
    private Function<Double, Double> activationFunction = null;
    private Function<Double, Double> activationFunctionPrime = null;

    public Neuron(
        Function<Double, Double> activationFunction,
        Function<Double, Double> activationFunctionPrime,
        NeuronType type
    ) {
        this.setType(type);
        this.setActivationFunction(activationFunction, activationFunctionPrime);
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

    public void setInput(Double input) throws NetworkException {
        this.input = input;
        this.activate();
    }

    public void setInput(List<Double> values, List<Edge> edges) throws NetworkException {
        Double sum = null;

        if (values.size() != edges.size()) {
            throw new NetworkException("Inequal values and weights");
        }

        sum = 0.0;
        for (int i = 0; i < values.size(); i++) {
            sum += values.get(i) * edges.get(i).getWeight();
        }

        this.setInput(sum);
    }

    public Double getOutput() {
        return this.output;
    }

    public void setOutput(Double output) {
        this.output = output;
    }

    public Double getDelta() {
        return this.delta;
    }

    public void setDelta(Double delta) {
        this.delta = delta;
    }

    public void activate() {
        if (this.getInput() == null) {
            return;
        }

        this.setOutput(this.getActivationFunction().apply(this.getInput()));
    }

    public Double getPrimeActivation() {
        if (this.getOutput() == null) {
            this.activate();
        }

        return this.getActivationFunctionPrime().apply(this.getOutput());
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

    public void computeDelta(List<Edge> edges) throws NetworkException {
        if (this.getOutput() == null || this.getInput() == null) {
            throw new NetworkException("Cannot calculate delta without input or output");
        }

        if (this.type != NeuronType.Hidden) {
            throw new NetworkException("Cannot use this formula on a non-hidden neuron");
        }

        Double sum = 0.0;
        for (Edge edge : edges) {
            sum += edge.getWeight() * edge.getDestination().getDelta();
        }

        this.setDelta(this.getPrimeActivation() * sum);
    }
}
