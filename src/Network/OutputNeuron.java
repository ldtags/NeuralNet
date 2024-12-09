package Network;

import java.util.List;
import java.util.ArrayList;

public class OutputNeuron extends BaseNeuron {
    private List<Edge> inputEdges = null;

    public OutputNeuron() {
        super(NeuronType.Output);
        this.inputEdges = new ArrayList<>();
    }

    public List<Edge> getInputEdges() {
        return this.inputEdges;
    }

    public void setInputEdges(List<Edge> inputEdges) {
        this.inputEdges = inputEdges;
    }

    public void activate() {
        if (this.getInput() == null) {
            return;
        }

        this.setOutput(logistic(this.getInput()));
    }

    public void computeDelta(Integer outputClass) {
        if (this.getInput() == null || this.getOutput() == null) {
            return;
        }

        this.setDelta(
            logisticPrime(this.getInput()) * (-2.0 * (outputClass - this.getOutput()))
        );
    }
}
