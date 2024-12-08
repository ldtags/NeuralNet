/*
 * Author: Liam D. Tangney
 */

package Network;

import java.util.List;

public class Neuron {
    private Double input = null;
    private Double output = null;
    private Double delta = null;
    private NeuronType type = null;

    public Neuron(NeuronType type) {
        this.setType(type);
    }

    public Neuron() {
        this(NeuronType.Hidden);
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

    public void activate() throws NetworkException {
        if (this.getInput() == null) {
            throw new NetworkException("Cannot activate neuron before input");
        }

        if (this.type == NeuronType.Input) {
            this.setOutput(this.getInput());
        } else {
            this.setOutput(1.0 / (1.0 + Math.exp(this.getInput())));
        }
    }

    public Double getPrimeActivation() throws NetworkException {
        if (this.getOutput() == null) {
            this.activate();
        }

        return this.getOutput() * (1 - this.getOutput());
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
