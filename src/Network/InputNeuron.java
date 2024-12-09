/*
 * Author: Liam D. Tangney
 */

package Network;

import java.util.List;
import java.util.ArrayList;

public class InputNeuron extends BaseNeuron {
    private List<Edge> outputEdges;

    public InputNeuron() {
        super(NeuronType.Input);
        this.outputEdges = new ArrayList<>();
    }

    public List<Edge> getOutputEdges() {
        return this.outputEdges;
    }

    public void setOutputEdges(List<Edge> outputEdges) {
        this.outputEdges = outputEdges;
    }

    public void activate() {
        this.setOutput(this.getInput());
    }
}
