/*
 * Author: Liam D. Tangney
 */

import java.util.List;
import java.util.ArrayList;

public class DataPoint {
    private List<Double> attributes = null;
    private List<Integer> targets = null;

    public DataPoint(List<Double> attributes, List<Integer> targets) {
        this.setAttributes(attributes);
        this.setTargets(targets);
    }

    public List<Double> getAttributes() {
        return this.attributes;
    }

    public void setAttributes(List<Double> attributes) {
        this.attributes = new ArrayList<>(attributes);
    }

    public List<Integer> getTargets() {
        return this.targets;
    }

    public void setTargets(List<Integer> targets) {
        this.targets = new ArrayList<>(targets);
    }
}
