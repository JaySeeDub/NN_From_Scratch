// Simple neuron object to fill a neural network

import java.util.Random;

public class Neuron {
    // Each neuron has weights, a bias, and a layer number
    // Default 1s, 0s for input
    double[] weightMatrix;
    double bias;
    int layer;

    public Neuron (int weight_size) {
        Random random = new Random();
        // weight number determined by network
            this.weightMatrix = new double[weight_size];
            this.bias = random.nextDouble(-1, 1);

    }

    // Manually set a neuron's weight vector
    public void setWeights(double[] weights) {
        this.weightMatrix = weights;
    }

    // Manually set a neuron's bias
    public void setBias(double bias) {
        this.bias = bias;
    }

    // Calculate the output of a neuron given an input matrix
    public double input(double[] inputMatrix) {
        double Z = Helper.dotProduct(inputMatrix, this.weightMatrix) + this.bias;
        return Helper.sigmoid(Z);
    }
}