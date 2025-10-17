// Test all functions of the network and train on test data

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
public class TestNetwork {
    public static void main(String[] args) {
        int inputSize = 784;
        int[] hiddenSize = {1, 30};
        int outputSize = 10;

        Network net = new Network(inputSize, hiddenSize, outputSize);

        // Load and process test data from csv
        String train_path = "C:\\Users\\chewy\\IdeaProjects\\NeuralNetwork\\src\\Data\\mnist_test.csv";
        String test_path  = "C:\\Users\\chewy\\IdeaProjects\\NeuralNetwork\\src\\Data\\mnist_test.csv";

        System.out.println(System.getProperty("user.dir"));

        // Data normalized, shuffled, and split into minibatches
        Helper.MNISTLoader.MNISTData train_data = Helper.MNISTLoader.loadMNIST(train_path, 60000, 30);
        Helper.MNISTLoader.MNISTData test_data = Helper.MNISTLoader.loadMNIST(test_path, 10000, 30);


        // Test forward pass
        double[][] outputs = net.forward(train_data.images[0][0]);

        // Test training
        System.out.println("\n+++++++++++++++ Training ++++++++++++++++");
        double eta = 10;
        int epochs = 6;
        net.train(train_data.images, train_data.labels, eta, epochs);
    }
}
