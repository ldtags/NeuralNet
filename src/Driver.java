/*
 * Author: Liam D. Tangney
 */

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.ArrayList;

import Agent.Agent;
import Agent.AgentException;

public class Driver {
    public static void main(String[] args) {
        List<Integer> hiddenLayers = new ArrayList<>();
        ArgumentIterator argIterator = new ArgumentIterator(args);
        String arg = null;
        File file = null;
        Agent agent = new Agent();

        while (argIterator.hasNext()) {
            switch (argIterator.nextFlag()) {
            case "-f":
                arg = argIterator.nextArgument();
                if (arg == null) {
                    System.err.println("-f must be followed by a value");
                    return;
                }

                file = new File(arg);
                if (!file.exists()) {
                    System.err.printf("%s does not exist\n", arg);
                    return;
                }

                if (!file.isFile()) {
                    System.err.printf("%s is not a file\n", arg);
                    return;
                }

                if (!file.canRead()) {
                    System.err.printf("%s cannot be read\n", arg);
                    return;
                }

                try {
                    agent.loadData(arg);
                } catch (IOException | NumberFormatException e) {
                    System.err.printf("An error occurred while parsing %s: %s", arg, e.getMessage());
                    return;
                }

                break;
            case "-h":
                while ((arg = argIterator.nextArgument()) != null) {
                    try {
                        hiddenLayers.add(Integer.parseInt(arg));
                    } catch (NumberFormatException e) {
                        System.err.printf("Invalid hidden layer: %s\n", arg);
                        return;
                    }
                }

                agent.setHiddenLayerSizes(hiddenLayers);
                break;
            case "-a":
                arg = argIterator.nextArgument();
                if (arg == null) {
                    System.err.println("-h must be followed by a value");
                    return;
                }

                try {
                    agent.setLearningRate(Double.parseDouble(arg));
                } catch (NumberFormatException e) {
                    System.err.printf("Invalid learning rate: %s\n", arg);
                    return;
                }

                break;
            case "-e":
                arg = argIterator.nextArgument();
                if (arg == null) {
                    System.err.println("-e must be followed by a value");
                    return;
                }

                try {
                    agent.setEpochLimit(Integer.parseInt(arg));
                } catch (NumberFormatException e) {
                    System.err.printf("Invalid epoch limit: %s\n", arg);
                    return;
                }

                break;
            case "-m":
                arg = argIterator.nextArgument();
                if (arg == null) {
                    System.err.println("-m must be followed by a value");
                    return;
                }

                try {
                    agent.setBatchSize(Integer.parseInt(arg));
                } catch (NumberFormatException e) {
                    System.err.printf("Invalid batch size: %s\n", arg);
                    return;
                }

                break;
            case "-l":
                arg = argIterator.nextArgument();
                if (arg == null) {
                    System.err.println("-l must be followed by a value");
                    return;
                }

                try {
                    agent.setRegularization(Double.parseDouble(arg));
                } catch (NumberFormatException e) {
                    System.err.printf("Invalid regularization hyperparameter: %s\n", arg);
                    return;
                }

                break;
            case "-r":
                agent.setRandomization(true);
                break;
            case "-w":
                arg = argIterator.nextArgument();
                if (arg == null) {
                    System.err.println("-w must be followed by a value");
                    return;
                }

                try {
                    agent.setWeightInitialization(Double.parseDouble(arg));
                } catch (NumberFormatException e) {
                    System.err.printf("Invalid weight initialization: %s\n", arg);
                    return;
                } catch (AgentException e) {
                    System.err.println(e.getMessage());
                    return;
                }

                break;
            case "-v":
                arg = argIterator.nextArgument();
                if (arg == null) {
                    System.err.println("-v must be followed by a value");
                    return;
                }

                try {
                    agent.setVerbosity(Integer.parseInt(arg));
                } catch (NumberFormatException e) {
                    System.err.printf("Invalid verbosity level: %s\n", arg);
                    return;
                }

                break;
            }
        }

        try {
            agent.start();
        } catch (AgentException e) {
            System.err.println(e.getMessage());
            return;
        }
    }
}
