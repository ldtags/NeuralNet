/*
 * Author: Liam D. Tangney
 */

package Network;

public abstract class BaseNeuron {
    private Double input = null;
    private Double output = null;
    private Double delta = null;
    private NeuronType type = null;

    public BaseNeuron(NeuronType type) {
        this.setType(type);
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

    public void setOutput(Double output) {
        this.output = output;
    }

    public Double getDelta() {
        return this.delta;
    }

    public void setDelta(Double delta) {
        this.delta = delta;
    }

    public static Double logistic(Double input) {
        return 1.0 / (1.0 + Math.exp(-1.0 * input));
    }

    public static Double logisticPrime(Double input) {
        Double logisticOutput = logistic(input);
        return logisticOutput * (1 - logisticOutput);
    }

    abstract public void activate();
}
