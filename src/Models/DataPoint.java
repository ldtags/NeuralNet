/*
 * Author: Liam D. Tangney
 */

package Models;

import java.util.List;
import java.util.ArrayList;

public class DataPoint {
    private List<Double> features = null;
    private List<Integer> outputClass = null;

    public DataPoint(List<Double> features, List<Integer> targets) {
        this.setFeatures(features);
        this.setOutputClass(targets);
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

    public List<Integer> getOutputClass() {
        return this.outputClass;
    }

    public Integer getDecodedOutputClass() {
        Integer index = 0;

        for (Integer target : this.outputClass) {
            if (target == 1) {
                return index + 1;
            }

            index++;
        }

        return 0;
    }

    public void setOutputClass(List<Integer> outputClass) {
        this.outputClass = new ArrayList<>(outputClass);
    }
}
