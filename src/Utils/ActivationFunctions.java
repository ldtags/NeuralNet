/*
 * Author: Liam D. Tangney
 */

package Utils;

public final class ActivationFunctions {
    /*
     * This class is essentially a namespace for neuron activation functions.
     * 
     * Because, Java.
     */

    public static Double logistic(Double input) {
        return 1.0 / (1.0 + Math.exp(-1.0 * input));
    }

    public static Double logisticPrime(Double input) {
        Double logisticOutput = logistic(input);
        return logisticOutput * (1.0 - logisticOutput);
    }

    public static Double reLU(Double input) {
        return Math.max(0, input);
    }

    public static Double reLUPrime(Double input) {
        if (input > 0) {
            return 1.0;
        }

        return 0.0;
    }

    public static Double softplus(Double input) {
        return Math.log(1.0 + Math.exp(input));
    }

    public static Double softplusPrime(Double input) {
        return logistic(input);
    }

    public static Double hyperbolicTangent(Double input) {
        return (
            Math.exp(input) - Math.exp(-1.0 * input)
        ) / (
            Math.exp(input) + Math.exp(-1.0 * input)
        );
    }

    public static Double hyperbolicTangentPrime(Double input) {
        return 1.0 - Math.pow(hyperbolicTangent(input), 2);
    }
}
