// Jesse Webb; CWID: 103-94-561
// 10/13/25
// Assignment 2.1
// Part 1 of building a neural net from scratch. Implements sigmoid neurons in a fully connected network
// with backpropagation and gradient descent on the simple test data of binary one hot vectors.
// Target output is just the flipped vector.

import java.util.Arrays;

public class Network {
    Neuron[] network;
    int input_size;
    int[] hidden_size;
    int output_size;

    public Network (int input_size, int[] hidden_size, int output_size) {
        network = new Neuron[input_size+hidden_size[0]*hidden_size[1]+output_size];
        this.input_size = input_size;
        this.hidden_size = hidden_size;
        this.output_size = output_size;

        // Fill network with neurons
        int neuron_index = 0;
        // Input neurons
        for (int i = 0; i < input_size; i++) {
            this.network[neuron_index] = new Neuron(0);
            neuron_index++;
        }
        // Number of hidden layers is index 0, number of neurons per layer index 1
        for (int i = 0; i < hidden_size[0]; i++) {
            for (int j = 0; j < hidden_size[1]; j++) {
                this.network[neuron_index] = new Neuron(1);
                neuron_index++;
            }
        }
        // Output neurons
        for (int i = 0; i < output_size; i++) {
            this.network[neuron_index] = new Neuron(2);
            neuron_index++;
        }
    }

    // Method for setting any weights in the network
    public void setWeights(int[] neuron_indices, double[][] values) {
        for (int i = 0; i < neuron_indices.length; i++) {
            this.network[neuron_indices[i]].setWeights(values[i]);
        }
    }

    // Method for setting any biases in the network
    public void setBiases(int[] neuron_indices, double[] values) {
        for (int i = 0; i < neuron_indices.length; i++) {
            this.network[neuron_indices[i]].setBias(values[i]);
        }
    }

    public void train(double[][][] inputs, double[][][] y, double eta, int epochs) {
        // Takes inputs as [mini_batch][Sample][Inputs]
        // y as [minibatch][Sample][Outputs]
        for (int epoch = 1; epoch <= epochs; epoch++) {
            System.out.println("--------------- Epoch (" + epoch + ") ---------------");
            for (int batch = 0; batch < inputs.length; batch++) {
                System.out.println("--------------- Batch (" + (batch+1) + ") ---------------");

                double[][][] GradVector1 = computeError(inputs[batch][0], y[batch][0]);
                double[][][] GradVector2 = computeError(inputs[batch][1], y[batch][1]);

                // Have to build the dimensions of GradVector to handle the different bias, weight shapes
                double[][][] GradVector = new double[GradVector1.length][][];

                for (int i = 0; i < GradVector1.length; i++) {
                    GradVector[i] = new double[GradVector1[i].length][];
                    for (int j = 0; j < GradVector1[i].length; j++) {
                        GradVector[i][j] = new double[GradVector1[i][j].length];

                        // And compute the average of each element
                        for (int k = 0; k < GradVector1[i][j].length; k++) {
                            GradVector[i][j][k] = (GradVector1[i][j][k] + GradVector2[i][j][k]) / 2.0;
                        }
                    }
                }

                System.out.println("Grad Vector in Training ============ " + Arrays.deepToString(GradVector));
                // backprop takes BiasGrad, WeightGrad, and lr
                backprop(GradVector[0], GradVector[1], eta);
            }
        }
    }

    public double [][] forward(double[] inputs) {
        // Takes input values and computes outputs for each layer of the network
        double[][] outputs = new double[3][];
        outputs[0] = new double[input_size];
        outputs[1] = new double[hidden_size[1]];
        outputs[2] = new double[output_size];

        // Layer 0 is just inputs (no computation needed)
        for (int i = 0; i < input_size; i++) {
            outputs[0][i] = inputs[i];
        }
        // Iterate through hidden layers to compute outputs
        int neuron_index = input_size;
        for (int i = 0; i < hidden_size[0]*hidden_size[1]; i++) {
            outputs[1][i] = this.network[neuron_index].input(inputs);
            neuron_index++;
        }
        // Final layer outputs
        for (int i = 0; i < output_size; i++) {
            outputs[2][i] = this.network[neuron_index].input(outputs[1]);
            neuron_index++;
        }

        System.out.println("\n================ Forward Pass ====================");
        System.out.println("\nInput: " + Arrays.toString(outputs[0]));
        System.out.println("Hidden Layer Output: " + Arrays.toString(outputs[1]));
        System.out.println("Final Layer Output: " + Arrays.toString(outputs[2]));
        return outputs;
    }

    public void backprop(double[][] biasGrad, double[][] weightGrad, double eta) {
        // Takes the biasGrad/weightGrad (pre-averaged) and learning rate
        // Modifies the weights and biases of each neuron by the given Grad vectors

        for (int i = 0; i < this.network.length; i++) {
            System.out.println("\n==================================================================");
            System.out.println("Neuron " + i + ": ");

            System.out.println("Initial Bias: " + this.network[i].bias);
            System.out.println("Initial Weights: " + Arrays.toString(this.network[i].weightMatrix));

            System.out.println("BiasGrad: " + Arrays.deepToString(biasGrad));
            System.out.println("WeightGrad: " + Arrays.deepToString(weightGrad));

            this.network[i].bias -= eta*biasGrad[i][0];
            for (int j = 0; j < this.network[i].weightMatrix.length; j++) {
                this.network[i].weightMatrix[j] -= eta * weightGrad[i][j];
            }
            System.out.println("Modified Bias: " + this.network[i].bias);
            System.out.println("Modified Weights: " + Arrays.toString(this.network[i].weightMatrix));
        }
    }

    public double [][][] computeError(double[] inputs, double[] y) {
        // Takes inputs, expected outputs, and network
        // Calls on hiddenError and outputError to generate GradVector for the whole network
        double[][][] GradVector = new double[2][this.network.length][];
        // Collect all network weights[neuron][weight_vector]
        //
        double[][] weights = new double[this.network.length][];
        int neuron_index = 0;
        // Input Layer
        for (int i = 0; i < input_size; i++) {
            weights[neuron_index] = this.network[neuron_index].weightMatrix;
            neuron_index++;
        }
        // Hidden Layers
        for (int i = 0; i < hidden_size[0]; i++) {
            for (int j = 0; j < hidden_size[1]; j++) {
                weights[neuron_index] = this.network[neuron_index].weightMatrix;
                neuron_index++;
            }
        }
        // Output Layer
        for (int i = 0; i < output_size; i++) {
            weights[neuron_index] = this.network[neuron_index].weightMatrix;
            neuron_index++;
        }

        // Now use those weights to calculate errors
        //
        double[][] outputs = this.forward(inputs);
        // GradVectorL = {biasGrad, weightGrad} for output layer
        double[][][] GradVectorL = this.outputError(outputs[2], outputs[1], y);
        // GradVectorHidden = {biasGrad, weightGrad} for hidden layers

        double[][][] GradVectorHidden = this.hiddenError(outputs[1], outputs[0], weights, GradVectorL[0]);


        System.out.println("\n============ Compute Errors ==============");
        System.out.println("\nFinal Layer Gradient");
        System.out.println("Bias: " + Arrays.deepToString(GradVectorL[0]));
        System.out.println("Weight: " + Arrays.deepToString(GradVectorL[1]));

        System.out.println("\nHidden Layer Gradient");
        System.out.println("Bias: " + Arrays.deepToString(GradVectorHidden[0]));
        System.out.println("Weight: " + Arrays.deepToString(GradVectorHidden[1]));

        // Now combine those errors into one vector of all gradient values
        // Input Layer
        int index = 0;
        for (int i = 0; i < input_size; i++) {
            GradVector[0][index] = new double[] {0};
            GradVector[1][index] = new double[] {0, 0, 0, 0};
            index++;
        }
        // Hidden Layers
        int hiddenIndex = 0;
        for (int i = 0; i < GradVectorHidden[1].length; i++) {
            GradVector[0][index] = GradVectorHidden[0][hiddenIndex];
            GradVector[1][index] = GradVectorHidden[1][hiddenIndex];
                index++;
                hiddenIndex++;
            }
        // Output Layer
        for (int i = 0; i < output_size; i++) {
            GradVector[0][index] = GradVectorL[0][i];
            GradVector[1][index] = GradVectorL[1][i];
            index++;
        }
        return GradVector;
    }

    public double[][][] outputError(double[] a, double[] aPrev, double[] y) {
        // Takes output, expected output, and output of L-1 layer
        // computes error on final layer as a vector, returning biasGrad, weightGrad
        double[][] biasGrad = new double[a.length][1];
        double[][] weightGrad = new double[a.length][aPrev.length];

        for (int j = 0; j < a.length; j++) {
            biasGrad[j][0] = (a[j] - y[j]) * a[j] * (1 - a[j]);
            // Times each previous a (WeightGradient)
            for (int i = 0; i < aPrev.length; i++)
                weightGrad[j][i] = biasGrad[j][0] * aPrev[i];
        }
        return new double[][][] {biasGrad, weightGrad};
    }

    public double[][][] hiddenError(double[] a, double[] aPrev, double[][] weights, double[][] deltaNext) {
        // Takes output of current layer, weights and errors of I+1 layer, and output of I-1 layer
        // Computes error on current layer as a vector, returning biasGrad, weightGrad

        double[][] biasGrad = new double[a.length][1];
        double[][] weightGrad = new double[a.length][aPrev.length];

        // For each ith hidden neuron,
        for (int i = 0; i < hidden_size[1]; i++) {
            // Reset sum
            double sum = 0.0;
            // For each output neuron,
            for (int j = 0; j < deltaNext.length; j++) {
                // Get that output neuron's error * its ith weight and add to sum
                sum += deltaNext[j][0] * weights[input_size+hidden_size[1]+j][i];
            }
            // Error equation for each neuron (BiasGradient) populates the ith neuron's biasGrad
            biasGrad[i][0] = sum * a[i] * (1 - a[i]);

            // Times each previous layer's a (WeightGradient) populates the ith neuron's kth weightGrad
            for (int k = 0; k < aPrev.length; k++) {
                weightGrad[i][k] = biasGrad[i][0] * aPrev[k];
            }
        }
        double [][][] out = new double[][][] {biasGrad, weightGrad};
        return out;
    }
}
