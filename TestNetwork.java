// Test all functions of the network and train on test data

import java.util.Arrays;

public class TestNetwork {
    public static void main(String[] args) {
        int inputSize = 4;
        int[] hiddenSize = {1, 3};
        int outputSize = 2;

        Network net = new Network(inputSize, hiddenSize, outputSize);

        // Indices of neurons to be set
        int[] indices = new int[net.network.length];
        for (int i = 0; i < net.network.length; i++) {
            indices[i] = i;
        }

        // Hardcoded weights and biases
        double[][] weights = {
                {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},
                {-0.21, 0.72, -0.25, 1}, {-0.94, -0.41, -0.47, 0.63}, {0.15, 0.55, -0.49, -0.75},
                {0.76, 0.48, -0.73},
                {0.34, 0.89, -0.23}
        };
        double[] biases = {0, 0, 0, 0, 0.1, -0.36, -0.31, 0.16, -0.46};

        System.out.println("Hardcoded Weights and Biases:");

        net.setWeights(indices, weights);
        net.setBiases(indices, biases);

        // Print hardcoded weights and biases
        for (int i = 0; i < net.network.length; i++) {
            System.out.print("(Node " + i + "): " + Arrays.toString(net.network[i].weightMatrix));
            System.out.print("; Bias = " + net.network[i].bias);
            System.out.println();
        }

        // Test data split into minibatches
        double[][][] inputs = {{
                {0, 1, 0, 1},
                {1, 0, 1, 0}
        },
                {{0, 0, 1, 1},
                {1, 1, 0, 0}}
        };

        // Expected output split into minibatches
        double[][][] y = {
                {{0, 1}, {1,0}},
                {{0, 1}, {1,0}}
        };

        // Test forward pass
        double[][] outputs = net.forward(inputs[0][0]);

        // Test training
        System.out.println("\n+++++++++++++++ Training ++++++++++++++++");
        double eta = 10;
        int epochs = 6;
        net.train(inputs, y, eta, epochs);
    }
}
