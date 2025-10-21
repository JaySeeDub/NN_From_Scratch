// Test all functions of the network and train on test data

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Scanner;

public class TestNetwork {
    public static void main(String[] args) throws IOException {
        System.out.println("Starting Neural Network Program...");

        // Initiate network
        int inputSize = 784;
        int[] hiddenSize = {1, 30};
        int outputSize = 10;

        Network net = new Network(inputSize, hiddenSize, outputSize);

        // Load and process test data from csv
        String train_path = System.getProperty("user.home") + "\\IdeaProjects\\NeuralNetwork\\src\\Data\\mnist_train.csv";
        String test_path = System.getProperty("user.home") + "\\IdeaProjects\\NeuralNetwork\\src\\Data\\mnist_test.csv";

        // Read commands from the user
        Scanner inputScanner = new Scanner(System.in);
        String command = " ";

        System.out.println("\nSet csv data paths? [y/n]");
        command = inputScanner.nextLine();

        if (Objects.equals(command, "y")) {
            System.out.println("Set train_path: ");
            train_path = inputScanner.nextLine();
            System.out.println("Set test_path: ");
            test_path = inputScanner.nextLine();
        }

        System.out.println("Loading train data from " + train_path + " ...");
        System.out.println("Loading test data from " + test_path + " ...");

        // Data normalized, shuffled, and split into minibatches
        Helper.MNISTLoader.MNISTData train_data = Helper.MNISTLoader.loadMNIST(train_path, 60000, 10);
        Helper.MNISTLoader.MNISTData test_data = Helper.MNISTLoader.loadMNIST(test_path, 10000, 10);


        boolean trained = false;
        while (!Objects.equals(command, "0")) {
            // Select command
            System.out.println("\n [1] TRAIN | [2] LOAD NETWORK | [3] TRAINING ACCURACY | [4] TEST ACCURACY | [5] RUN TEST | [6] SHOW MISCLASSIFIED | [7] SAVE | [0] EXIT");
            command = inputScanner.nextLine();

            System.out.println();
            // Training
            if (Objects.equals(command, "1")) {
                double eta = 10.0;
                int epochs = 1;

                System.out.println("Enter learning rate: ");
                eta = inputScanner.nextDouble();
                System.out.println("Enter number of epochs: ");
                epochs = inputScanner.nextInt();

                System.out.println("\n+++++++++++++++ Training ++++++++++++++++");
                System.out.println();

                net.train(train_data.images, train_data.labels, eta, epochs);

                trained = true;
            }

            else if (Objects.equals(command, "2")) {
                System.out.println("Enter load_path: ");
                String load_path = inputScanner.nextLine();
                Helper.loadNetwork(net,load_path);

                System.out.println("\nSuccessfully loaded network state from " + System.getProperty("user.dir") + "\\" + load_path + "!");
                trained = true;
            }

            else if (Objects.equals(command, "3")) {
                if (trained) {
                    System.out.println("Displaying network accuracy on training data...\n");
                    net.testNetwork(train_data.images, train_data.labels, false, false);
                }
                else {
                    System.out.println("No model trained or loaded...\n");
                }
            }

            else if (Objects.equals(command, "4")) {
                if (trained) {
                    System.out.println("Displaying network accuracy on testing data...\n");
                    net.testNetwork(test_data.images, test_data.labels, false, false);
                }
                else {
                    System.out.println("No model trained or loaded...\n");
                }
            }

            else if (Objects.equals(command, "5")) {
                if (trained) {
                    System.out.println("Displaying testing images...\n");
                    net.testNetwork(test_data.images, test_data.labels, true, false);
                }
                else {
                    System.out.println("No model trained or loaded...\n");
                }
            }

            else if (Objects.equals(command, "6")) {
                if (trained) {
                    System.out.println("Displaying testing images...\n");
                    net.testNetwork(test_data.images, test_data.labels, true, true);
                }
                else {
                    System.out.println("No model trained or loaded...\n");
                }
            }

            else if (Objects.equals(command, "7")) {
                if (trained) {
                    System.out.println("Enter save_path: ");
                    String save_path = inputScanner.nextLine();
                    Helper.saveNetwork(net, save_path);

                    System.out.println("\nSuccessfully saved network state to " + System.getProperty("user.dir") + "\\" + save_path + "!");
                }
                else {
                    System.out.println("No model trained or loaded...\n");
                }
            }

            else System.out.println("Not a valid command...");
        }
        System.out.println("Exiting Program...\n");
    }
}