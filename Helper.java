// Helper functions for the network
import java.io.*;
import java.util.*;

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
    public static double sigmoid(double Z) {
        return 1 / (1 + Math.exp(-Z));
    }


    // For handling csv read/write
    public static class MNISTLoader {

        public static class MNISTData {
            public double[][] images;  // [numSamples][784]
            public double[][] labels;  // [numSamples][10]
        }

        public static MNISTData loadMNIST(String filePath, int numSamples) {
            List<double[]> imageList = new ArrayList<>();
            List<double[]> digitList = new ArrayList<>();

            String line;
            String delimiter = ",";

            try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
                int count = 0;
                while ((line = br.readLine()) != null && count < numSamples) {
                    String[] values = line.split(delimiter);

                    // Get label
                    int label = Integer.parseInt(values[0]);

                    // One-hot encoding
                    double[] oneHot = new double[10];
                    oneHot[label] = 1.0;

                    // Read and cale pixel values
                    double[] pixels = new double[784];
                    for (int i = 1; i < values.length; i++) {
                        pixels[i - 1] = Double.parseDouble(values[i]) / 255.0;
                    }

                    imageList.add(pixels);
                    digitList.add(oneHot);
                    count++;
                }
            } catch (IOException e) {
                e.printStackTrace();
            }

            MNISTData data = new MNISTData();
            double[][] orderedImages = imageList.toArray(new double[0][0]);
            double[][] orderedLabels = digitList.toArray(new double[0][0]);
            data.images = new double[orderedLabels.length][orderedLabels[0].length];
            data.labels = new double[orderedLabels.length][orderedLabels[0].length];

            // Shuffle the data entries
            // Create list of shuffled indices
            ArrayList<Integer> indices = new ArrayList<>();
            for (int i = 0; i < orderedLabels.length; i++) indices.add(i);
            Collections.shuffle(indices);

            // Shuffle fill data.images and data.labels with the ordered data
            for (int i = 0; i < orderedLabels.length; i++) {
                data.images[i] = orderedImages[indices.get(i)];
                data.labels[i] = orderedLabels[indices.get(i)];
            }
            return data;
        }
    }
    // save a network's parameters to csv
    public static void saveNetwork(Network net, String path) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
            bw.write(String.format("%d,%d,%d\n", net.input_size, net.hidden_size[0], net.output_size));
            for (int i = 0; i < net.network.length; i++) {
                double bias = net.network[i].bias;
                bw.write(Double.toString(bias));
                for (double weight : net.network[i].weightMatrix) {
                    bw.write("," + Double.toString(weight));
                }
                bw.write("\n");
            }
        }
    }
    // load network weights into an existing network
    public static void loadNetwork(Network net, String path) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            br.readLine();
            String line;
            int i = 0;
            while ((line = br.readLine()) != null && i < net.network.length) {
                String[] content = line.split(",");
                net.network[i].bias = Double.parseDouble(content[0]);
                double[] weight = new double[content.length - 1];
                for (int j = 1; j < content.length; j++) weight[j-1] = Double.parseDouble(content[j]);
                net.network[i].weightMatrix = weight;
                i++;
            }
        }
    }
    public static void showAscii(double[] pixels) {
        for (int r = 0; r < 28; r++) {
            StringBuilder sb = new StringBuilder();
            for (int c = 0; c < 28; c++) {
                double v = pixels[r*28 + c];
                char ch = v > 0.9 ? '@' : v > 0.75 ? '#' : v > 0.6 ? '+' :  v > 0.4 ? 'o' : v > 0.1 ? '.' : ' ';
                sb.append(ch);
            }
            System.out.println(sb.toString());
        }
    }
}