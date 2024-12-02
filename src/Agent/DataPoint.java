/*
 * Author: Liam D. Tangney
 */

package Agent;

import java.util.List;
import java.util.ArrayList;

public class DataPoint {
    private List<Double> features = null;
    private List<Integer> targets = null;

    public DataPoint(List<Double> features, List<Integer> targets) {
        this.setFeatures(features);
        this.setTargets(targets);
    }

    public List<Double> getFeatures() {
        return this.features;
    }

    public void setFeatures(List<Double> features) {
        this.features = new ArrayList<>(features);
    }

    public void setFeature(Integer index, Double feature) {
        this.features.set(index, feature);
    }

    public List<Integer> getTargets() {
        return this.targets;
    }

    public void setTargets(List<Integer> targets) {
        this.targets = new ArrayList<>(targets);
    }
}
