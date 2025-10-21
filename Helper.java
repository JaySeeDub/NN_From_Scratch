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
            public double[][][] images;  // [batches][batchSize][784]
            public double[][][] labels;  // [batches][batchSize][10]
        }

        public static MNISTData loadMNIST(String filePath, int numSamples, int batch_size) {
            List<double[]> imageList = new ArrayList<>();
            List<double[]> digitList = new ArrayList<>();

            String line;
            String delimiter = ",";

            try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
                int count = 0;
                while ((line = br.readLine()) != null && count < numSamples) {
                    String[] values = line.split(delimiter);

                    // Parse label
                    int label = Integer.parseInt(values[0]);

                    // One-hot encode label (10 outputs)
                    double[] oneHot = new double[10];
                    oneHot[label] = 1.0;

                    // Parse pixel values and scale to 0–1
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
            double[][] shuffledImages = new double[orderedLabels.length][orderedLabels[0].length];
            double[][] shuffledLabels = new double[orderedLabels.length][orderedLabels[0].length];

            // Shuffle the data entries
            // Create list of shuffled indices
            ArrayList<Integer> indices = new ArrayList<>();
            for (int i = 0; i < orderedLabels.length; i++) indices.add(i);
            Collections.shuffle(indices);

            // Shuffle fill data.images with the ordered data
            for (int i = 0; i < orderedLabels.length; i++) {
                shuffledImages[i] = orderedImages[indices.get(i)];
                shuffledLabels[i] = orderedLabels[indices.get(i)];
            }

            // Minibatch number
            int num_batches = (int) Math.ceil((double) orderedLabels.length / batch_size);

            // Initialize data variables with shape [num_batches][batch_size][data_size]
            data.images = new double[num_batches][batch_size][orderedImages[0].length];
            data.labels = new double[num_batches][batch_size][orderedLabels[0].length];

            // Split shuffled data into minibatches
            for (int batch_num = 0; batch_num < num_batches - 1; batch_num++) {
                // For each minibatch, fill with shuffled data
                for (int i = 0; i < batch_size; i++) {
                    int j = batch_num * batch_size + i;
                    data.images[batch_num][i] = shuffledImages[j];
                    data.labels[batch_num][i] = shuffledLabels[j];
                }
            }
            return data;
        }
    }
        // save a network's parameters to csv
        public static void saveNetwork(Network net, String path) throws IOException {
            try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
                bw.write(String.format("%d,%d,%d\n", net.input_size, net.hidden_size[0], net.output_size));
                for (int i = 0; i < net.network.length; i++) {
                    double b = net.network[i].bias;
                    bw.write(Double.toString(b));
                    for (double w : net.network[i].weightMatrix) {
                        bw.write("," + Double.toString(w));
                    }
                    bw.write("\n");
                }
            }
        }
        // load network weights into an existing network
        public static void loadNetwork(Network net, String path) throws IOException {
            try (BufferedReader br = new BufferedReader(new FileReader(path))) {
                String header = br.readLine();
                String line;
                int idx = 0;
                while ((line = br.readLine()) != null && idx < net.network.length) {
                    String[] tok = line.split(",");
                    net.network[idx].bias = Double.parseDouble(tok[0]);
                    double[] w = new double[tok.length - 1];
                    for (int j = 1; j < tok.length; j++) w[j-1] = Double.parseDouble(tok[j]);
                    net.network[idx].weightMatrix = w;
                    idx++;
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