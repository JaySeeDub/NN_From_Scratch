// Helper functions for the network

public class Helper {

    // Dot product matrix multiplication loop
    public static double dotProduct(double[] vectorA, double[] vectorB) {
        double sum = 0;
        for (int i = 0; i < vectorA.length; i++) {
            sum += vectorA[i] * vectorB[i];
        }
        return sum;
    }

    // The sigmoid activation function
    public static double sigmoid(double Z){
        return 1 / (1 + Math.exp(-Z));
        }
    }
