// Jesse Webb; CWID: 103-94-561
// 10/21/25
// Assignment 2.2
// Part 2 of building a neural net from scratch. Added interactive network options and functionality
// to train digit recognition on the MNIST dataset

import java.util.Arrays;
import java.util.Objects;
import java.util.Scanner;

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

        // Fill network with neurons and instantiate random weights / biases
        int neuron_index = 0;
        // Input neurons
        for (int i = 0; i < input_size; i++) {
            this.network[neuron_index] = new Neuron(0);
            neuron_index++;
        }
        // Number of hidden layers is index 0, number of neurons per layer index 1
        for (int i = 0; i < hidden_size[0]; i++) {
            // If not first hidden layer, hidden_size number of weights
            if (hidden_size[0] != 1 && i > 0) {
                for (int j = 0; j < hidden_size[1]; j++) {
                    this.network[neuron_index] = new Neuron(hidden_size[1]);
                    neuron_index++;
                }
            }
            // First hidden layer has inputs number of weights
            else for (int j = 0; j < hidden_size[1]; j++) {
                this.network[neuron_index] = new Neuron(input_size);
                neuron_index++;
            }
        }
        // Output neurons
        for (int i = 0; i < output_size; i++) {
            this.network[neuron_index] = new Neuron(hidden_size[1]);
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

            // Have to build the dimensions of GradVector to handle the different bias, weight shapes
            double[][][] GradVectorTemp = computeError(inputs[0][0], y[0][0]);
            double[][][] GradVector = new double[GradVectorTemp.length][][];

            // initialize with all zeros
            for (int i = 0; i < GradVectorTemp.length; i++) {
                GradVector[i] = new double[GradVectorTemp[i].length][];
                for (int j = 0; j < GradVectorTemp[i].length; j++) {
                    GradVector[i][j] = new double[GradVectorTemp[i][j].length];
                }
            }

            // Loop over all batches and average the Gradients
            for (int batch = 0; batch < inputs.length; batch++) {
                GradVectorTemp = computeError(inputs[batch][0], y[batch][0]);
                for (int i = 0; i < GradVectorTemp.length; i++) {
                    for (int j = 0; j < GradVectorTemp[i].length; j++) {
                        for (int k = 0; k < GradVectorTemp[i][j].length; k++) {
                            GradVector[i][j][k] += GradVectorTemp[i][j][k] / inputs.length;
                        }
                    }
                }
            }
            // backprop takes BiasGrad, WeightGrad, and lr
            backprop(GradVector[0], GradVector[1], eta);

            // Print results from a forward pass on the training data, false means no displayed images
            testNetwork(inputs, y, false, false);
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
        return outputs;
    }

    public void backprop(double[][] biasGrad, double[][] weightGrad, double eta) {
        // Takes the biasGrad/weightGrad (pre-averaged) and learning rate
        // Modifies the weights and biases of each neuron by the given Grad vectors

        for (int i = 0; i < this.network.length; i++) {
            this.network[i].bias -= eta*biasGrad[i][0];
            for (int j = 0; j < this.network[i].weightMatrix.length; j++) {
                this.network[i].weightMatrix[j] -= eta * weightGrad[i][j];
            }
        }
    }

    public double [][][] computeError(double[] inputs, double[] y) {
        // Takes inputs, expected outputs, and network
        // Calls on hiddenError and outputError to generate GradVector for the whole network

        double[][][] GradVector = new double[2][this.network.length][];

        // Collect all network weights[neuron][weight_vector]
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

//        System.out.println("Outputs 1: " + Arrays.toString(outputs[1]));
//        System.out.println("Outputs 0: " + Arrays.toString(outputs[0]));
//        System.out.println("weights: " + Arrays.deepToString(weights));
//        System.out.println("GradVector: " + Arrays.deepToString(GradVectorL[0]));

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
                sum += deltaNext[j][0] * weights[input_size + hidden_size[0] * hidden_size[1] + j][i];
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

    public void testNetwork(double[][][] inputs, double[][][] labels, Boolean display, Boolean misclassified) {
        int[] digit_count = new int[10];
        int[] correct_count = new int[10];
        String command = "";

        // Iterate through each image in each batch of the dataset
        for (int batch = 0; batch < inputs.length; batch++) {
            int i = 0;
            for (double[] image : inputs[batch]) {
                // For each image, get the true digit label
                double[] label_vector = labels[batch][i];
                int true_label = 0;
                for (int j = 0; j < label_vector.length; j++) {
                    if (label_vector[j] == 1.0) {
                        true_label = j;
                        break;
                    }
                }
                // Count the true digit label
                digit_count[true_label]++;

                // Return all neuron outputs on that image
                double[][] outputs = forward(image);
                // Get the output layer
                double[] preds = outputs[2];

                // Get the most confident digit prediction
                double conf = 0;
                int pred_label = 0;
                for (int j = 0; j < preds.length; j++) {
                    if (preds[j] > conf) {
                        conf = preds[j];
                        pred_label = j;
                    }
                }
                // If correct prediction, add correct count to that digit
                if (pred_label == true_label) {
                    correct_count[pred_label]++;
                }
                // If incorrect prediction and display mode, show each incorrect image
                else if (display && misclassified) {
                    i++;
                    // If display mode, initialize a scanner
                    Scanner inputScanner = new Scanner(System.in);

                    System.out.println("Misclassified Image " + i + ":  \n");
                    Helper.showAscii(image);
                    System.out.println("Predicted: " + pred_label + "  ;  Actual: " + true_label);
                    System.out.println("\nContinue? [y/n]");
                    command = inputScanner.nextLine();

                    if (Objects.equals(command, "n")) {
                        break;
                    }
                }
                // In display mode, show each image, its label, its prediction, and give an option to break
                if (display && !misclassified) {
                    i++;
                    // If display mode, initialize a scanner
                    Scanner inputScanner = new Scanner(System.in);

                    System.out.println("Image " + i + ":  \n");
                    Helper.showAscii(image);
                    System.out.println("Predicted: " + pred_label + "  ;  Actual: " + true_label);
                    System.out.println("\nContinue? [y/n]");
                    command = inputScanner.nextLine();
                    }
                if (Objects.equals(command, "n")) {
                    break;
                }
            }
            if (Objects.equals(command, "n")) {
                break;
            }
        }

        // Print out correct digit counts / total digit counts
        System.out.println("=================== Accuracies ===================");
        int totalCorrect = 0;
        int totalDigits = 0;
        for (int digit = 0; digit < 10; digit++) {
            double acc = (100 * (double) correct_count[digit] /digit_count[digit]);
            System.out.print(digit + ": " + correct_count[digit] + "/" + digit_count[digit] + " = " + String.format("%.3f",acc) + "  ;  ");
            totalCorrect += correct_count[digit];
            totalDigits += digit_count[digit];
        }
        double percent = (100.0 * totalCorrect / totalDigits);
        System.out.println("\nTotal Accuracy:  " + totalCorrect + "/" + totalDigits + " = " + String.format("%.3f",percent));
    }
}