#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cassert>
#include <string>

using namespace std;

double relu(double x) {
    return x > 0 ? x : 0;
}

double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

int readInt(ifstream &ifs) {
    unsigned char bytes[4];
    ifs.read((char*)bytes, 4);
    return ((int)bytes[0] << 24) | ((int)bytes[1] << 16) | ((int)bytes[2] << 8) | ((int)bytes[3]);
}

vector<vector<double>> read_mnist_image(ifstream &ifs, int rows, int cols) {
    vector<vector<double>> image(rows, vector<double>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            unsigned char pixel = 0;
            ifs.read((char*)&pixel, sizeof(pixel));
            // Normalize pixel to [0,1]
            image[i][j] = pixel / 255.0;
        }
    }
    return image;
}

vector<vector<vector<double>>> load_mnist_images(const string &filename) {
    ifstream ifs(filename, ios::binary);
    if (!ifs) {
        cerr << "Could not open file " << filename << endl;
        exit(1);
    }
    int magic = readInt(ifs);
    int num_images = readInt(ifs);
    int rows = readInt(ifs);
    int cols = readInt(ifs);
    vector<vector<vector<double>>> images;
    for (int i = 0; i < num_images; i++) {
        images.push_back(read_mnist_image(ifs, rows, cols));
    }
    return images;
}

vector<int> load_mnist_labels(const string &filename) {
    ifstream ifs(filename, ios::binary);
    if (!ifs) {
        cerr << "Could not open file " << filename << endl;
        exit(1);
    }
    int magic = readInt(ifs);
    int num_labels = readInt(ifs);
    vector<int> labels(num_labels);
    for (int i = 0; i < num_labels; i++) {
        unsigned char temp = 0;
        ifs.read((char*)&temp, sizeof(temp));
        labels[i] = temp;
    }
    return labels;
}

vector<double> softmax(const vector<double> &x) {
    double max_val = *max_element(x.begin(), x.end());
    vector<double> exp_x(x.size());
    double sum = 0;
    for (size_t i = 0; i < x.size(); i++) {
        exp_x[i] = exp(x[i] - max_val);
        sum += exp_x[i];
    }
    for (size_t i = 0; i < x.size(); i++) {
        exp_x[i] /= sum;
    }
    return exp_x;
}

double cross_entropy_loss(const vector<double> &pred, int label) {
    double epsilon = 1e-12;
    return -log(pred[label] + epsilon);
}

vector<double> d_softmax_cross_entropy(const vector<double> &pred, int label) {
    vector<double> grad = pred;
    grad[label] -= 1.0;
    return grad;
}

vector<double> flatten(const vector<vector<vector<double>>> &input) {
    vector<double> output;
    for (size_t c = 0; c < input.size(); c++) {
        for (size_t i = 0; i < input[c].size(); i++) {
            for (size_t j = 0; j < input[c][i].size(); j++) {
                output.push_back(input[c][i][j]);
            }
        }
    }
    return output;
}

vector<vector<vector<double>>> unflatten(const vector<double> &input, int channels, int height, int width) {
    vector<vector<vector<double>>> output(channels, vector<vector<double>>(height, vector<double>(width, 0.0)));
    int index = 0;
    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                output[c][i][j] = input[index++];
            }
        }
    }
    return output;
}

// --- Folding operation ---
// This function implements a vertical fold that splits the image into top and bottom halves,
// then concatenates them along the width dimension. For a HxW image, the output is (H/2)x(2W).
vector<vector<double>> foldImageVertical(const vector<vector<double>> &image) {
    int H = image.size();
    int W = image[0].size();
    assert(H % 2 == 0); // For simplicity, we assume H is even.
    int newH = H / 2;
    int newW = W * 2;
    vector<vector<double>> folded(newH, vector<double>(newW, 0.0));
    for (int i = 0; i < newH; i++) {
        // Copy top half into the first W columns.
        for (int j = 0; j < W; j++) {
            folded[i][j] = image[i][j];
        }
        // Copy bottom half into the next W columns.
        for (int j = 0; j < W; j++) {
            folded[i][j + W] = image[i + newH][j];
        }
    }
    return folded;
}

class ConvLayer {
public:
    int num_filters, filter_size;
    double learning_rate;
    vector<vector<vector<double>>> filters;
    vector<vector<vector<double>>> last_input;
    vector<vector<vector<double>>> last_conv;

    // For multi-channel support (if needed) you could extend this class.
    ConvLayer(int num_filters, int filter_size, double lr)
        : num_filters(num_filters), filter_size(filter_size), learning_rate(lr) {
        filters.resize(num_filters, vector<vector<double>>(filter_size, vector<double>(filter_size)));
        srand(time(0));
        for (int f = 0; f < num_filters; f++) {
            for (int i = 0; i < filter_size; i++) {
                for (int j = 0; j < filter_size; j++) {
                    filters[f][i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.2;
                }
            }
        }
    }

    // Forward pass: input is a 2D image.
    // Output is a 3D volume: [num_filters][out_height][out_width]
    vector<vector<vector<double>>> forward(const vector<vector<double>> &input) {
        last_input.clear();
        last_input.push_back(input);
        int h = input.size();
        int w = input[0].size();
        int out_h = h - filter_size + 1;
        int out_w = w - filter_size + 1;
        vector<vector<vector<double>>> output(num_filters, vector<vector<double>>(out_h, vector<double>(out_w, 0.0)));
        last_conv = output; // initialize same shape

        for (int f = 0; f < num_filters; f++) {
            for (int i = 0; i < out_h; i++) {
                for (int j = 0; j < out_w; j++) {
                    double sum = 0.0;
                    for (int r = 0; r < filter_size; r++) {
                        for (int c = 0; c < filter_size; c++) {
                            sum += input[i + r][j + c] * filters[f][r][c];
                        }
                    }
                    last_conv[f][i][j] = sum;
                    output[f][i][j] = relu(sum);
                }
            }
        }
        return output;
    }

    vector<vector<double>> backward(const vector<vector<vector<double>>> &d_out) {
        const vector<vector<double>> &input = last_input[0];
        int h = input.size();
        int w = input[0].size();
        int out_h = h - filter_size + 1;
        int out_w = w - filter_size + 1;
        vector<vector<double>> d_input(h, vector<double>(w, 0.0));
        vector<vector<vector<double>>> d_filters(num_filters, vector<vector<double>>(filter_size, vector<double>(filter_size, 0.0)));

        for (int f = 0; f < num_filters; f++) {
            for (int i = 0; i < out_h; i++) {
                for (int j = 0; j < out_w; j++) {
                    double d_relu = relu_derivative(last_conv[f][i][j]);
                    double delta = d_out[f][i][j] * d_relu;
                    for (int r = 0; r < filter_size; r++) {
                        for (int c = 0; c < filter_size; c++) {
                            d_filters[f][r][c] += input[i + r][j + c] * delta;
                            d_input[i + r][j + c] += filters[f][r][c] * delta;
                        }
                    }
                }
            }
        }
        for (int f = 0; f < num_filters; f++) {
            for (int i = 0; i < filter_size; i++) {
                for (int j = 0; j < filter_size; j++) {
                    filters[f][i][j] -= learning_rate * d_filters[f][i][j];
                }
            }
        }
        return d_input;
    }
};

class MaxPoolLayer {
public:
    int pool_size;
    vector<vector<vector<int>>> max_indices;
    vector<vector<vector<double>>> last_input;

    MaxPoolLayer(int pool_size) : pool_size(pool_size) {}

    vector<vector<vector<double>>> forward(const vector<vector<vector<double>>> &input) {
        last_input = input;
        int channels = input.size();
        int h = input[0].size();
        int w = input[0][0].size();
        int out_h = h / pool_size;
        int out_w = w / pool_size;
        vector<vector<vector<double>>> output(channels, vector<vector<double>>(out_h, vector<double>(out_w, 0.0)));
        max_indices.clear();
        max_indices.resize(channels, vector<vector<int>>(out_h, vector<int>(out_w, 0)));

        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < out_h; i++) {
                for (int j = 0; j < out_w; j++) {
                    double max_val = -1e9;
                    int max_index = 0;
                    for (int r = 0; r < pool_size; r++) {
                        for (int s = 0; s < pool_size; s++) {
                            int cur_i = i * pool_size + r;
                            int cur_j = j * pool_size + s;
                            double val = input[c][cur_i][cur_j];
                            int index = r * pool_size + s;
                            if (val > max_val) {
                                max_val = val;
                                max_index = index;
                            }
                        }
                    }
                    output[c][i][j] = max_val;
                    max_indices[c][i][j] = max_index;
                }
            }
        }
        return output;
    }

    vector<vector<vector<double>>> backward(const vector<vector<vector<double>>> &d_out) {
        int channels = last_input.size();
        int h = last_input[0].size();
        int w = last_input[0][0].size();
        int out_h = h / pool_size;
        int out_w = w / pool_size;
        vector<vector<vector<double>>> d_input(channels, vector<vector<double>>(h, vector<double>(w, 0.0)));

        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < out_h; i++) {
                for (int j = 0; j < out_w; j++) {
                    int max_index = max_indices[c][i][j];
                    int r = max_index / pool_size;
                    int s = max_index % pool_size;
                    int cur_i = i * pool_size + r;
                    int cur_j = j * pool_size + s;
                    d_input[c][cur_i][cur_j] = d_out[c][i][j];
                }
            }
        }
        return d_input;
    }
};

class FCLayer {
public:
    int input_size, output_size;
    double learning_rate;
    vector<vector<double>> weights;
    vector<double> biases;
    vector<double> last_input;

    FCLayer(int input_size, int output_size, double lr)
        : input_size(input_size), output_size(output_size), learning_rate(lr) {
        weights.resize(output_size, vector<double>(input_size));
        biases.resize(output_size, 0.0);
        srand(time(0));
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                weights[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
            }
        }
    }

    vector<double> forward(const vector<double> &input) {
        last_input = input;
        vector<double> output(output_size, 0.0);
        for (int i = 0; i < output_size; i++) {
            double sum = biases[i];
            for (int j = 0; j < input_size; j++) {
                sum += weights[i][j] * input[j];
            }
            output[i] = sum;
        }
        return output;
    }

    vector<double> backward(const vector<double> &d_out) {
        vector<double> d_input(input_size, 0.0);
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                d_input[j] += weights[i][j] * d_out[i];
                weights[i][j] -= learning_rate * d_out[i] * last_input[j];
            }
            biases[i] -= learning_rate * d_out[i];
        }
        return d_input;
    }
};

int main() {
    srand(time(0));

    cout << "Loading MNIST dataset..." << endl;
    vector<vector<vector<double>>> train_images = load_mnist_images("train-images.idx3-ubyte");
    vector<int> train_labels = load_mnist_labels("train-labels.idx1-ubyte");
    cout << "Loaded " << train_images.size() << " training images." << endl;

    // --- Preprocessing: Folding ---
    // We perform a vertical fold on each image.
    // For a 28x28 image, the folded image becomes 14x56.
    vector<vector<vector<double>>> folded_images;
    for (size_t i = 0; i < train_images.size(); i++) {
        // Since MNIST images are single-channel, extract the 2D image:
        vector<vector<double>> image = train_images[i];
        vector<vector<double>> folded = foldImageVertical(image);
        folded_images.push_back(folded);
    }

    // --- Build the network ---
    // After folding:
    // Input: 14x56 image (single channel)
    // Convolution: 8 filters of size 3x3 → output size: 14-3+1 = 12 (height) x (56-3+1 = 54 width) → [8 x 12 x 54]
    // Max Pooling: 2x2 → output size: 12/2 = 6 x 54/2 = 27 → [8 x 6 x 27]
    // Flatten: 8 * 6 * 27 = 1296
    // Fully Connected Layer: 1296 → 10 (class scores)
    double conv_lr = 0.001;
    double fc_lr = 0.001;
    int num_filters = 8;
    int filter_size = 3;
    ConvLayer conv(num_filters, filter_size, conv_lr);
    MaxPoolLayer pool(2);
    int fc_input_size = num_filters * 6 * 27; // 8 * 6 * 27 = 1296
    int num_classes = 10;
    FCLayer fc(fc_input_size, num_classes, fc_lr);

    int epochs = 30;  
    int num_samples = folded_images.size();
    double total_loss = 0.0;
    int correct = 0;

    cout << "Starting training..." << endl;
    for (int epoch = 0; epoch < epochs; epoch++) {
        total_loss = 0.0;
        correct = 0;
        for (int idx = 0; idx < num_samples; idx++) {
            vector<vector<double>> image = folded_images[idx];
            int label = train_labels[idx];

            vector<vector<vector<double>>> conv_out = conv.forward(image);
            vector<vector<vector<double>>> pool_out = pool.forward(conv_out);
            vector<double> flat = flatten(pool_out);
            vector<double> fc_out = fc.forward(flat);
            vector<double> probs = softmax(fc_out);

            double loss = cross_entropy_loss(probs, label);
            total_loss += loss;
            int predicted = distance(probs.begin(), max_element(probs.begin(), probs.end()));
            if (predicted == label)
                correct++;

            vector<double> d_loss = d_softmax_cross_entropy(probs, label); // Gradient at fc layer output
            vector<double> d_fc = fc.backward(d_loss);  // Backprop through fully connected layer
            vector<vector<vector<double>>> d_pool = unflatten(d_fc, num_filters, 6, 27);
            vector<vector<vector<double>>> d_conv = pool.backward(d_pool);
            conv.backward(d_conv);
        }
        double avg_loss = total_loss / num_samples;
        double accuracy = (double)correct / num_samples * 100;
        cout << "Epoch " << epoch + 1 << ": Average Loss = " << avg_loss << ", Accuracy = " << accuracy << "%" << endl;
    }

    cout << "Training complete." << endl;
    return 0;
}
