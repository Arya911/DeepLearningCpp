import time
import numpy as np
import struct

# Load MNIST Dataset
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols) / 255.0
    return images

def load_labels(filename):
    with open(filename, 'rb') as f:
        _, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Convolutional Layer
class ConvLayer:
    def __init__(self, num_filters, filter_size, learning_rate):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.learning_rate = learning_rate
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.1

    def forward(self, input_image):
        self.input = input_image
        h, w = input_image.shape
        output_dim = h - self.filter_size + 1
        self.output = np.zeros((self.num_filters, output_dim, output_dim))
        
        for f in range(self.num_filters):
            for i in range(output_dim):
                for j in range(output_dim):
                    region = input_image[i:i+self.filter_size, j:j+self.filter_size]
                    self.output[f, i, j] = np.sum(region * self.filters[f])
        
        return self.output

    def backward(self, d_output):
        d_filters = np.zeros_like(self.filters)
        for f in range(self.num_filters):
            for i in range(d_output.shape[1]):
                for j in range(d_output.shape[2]):
                    region = self.input[i:i+self.filter_size, j:j+self.filter_size]
                    d_filters[f] += d_output[f, i, j] * region
        self.filters -= self.learning_rate * d_filters  # Update weights

# Max Pooling Layer
class MaxPoolLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, input_volume):
        self.input = input_volume
        depth, h, w = input_volume.shape
        output_dim = h // self.pool_size
        self.output = np.zeros((depth, output_dim, output_dim))

        for d in range(depth):
            for i in range(output_dim):
                for j in range(output_dim):
                    region = input_volume[d, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size]
                    self.output[d, i, j] = np.max(region)
        
        return self.output

    def backward(self, d_output):
        d_input = np.zeros_like(self.input)
        depth, h, w = self.input.shape
        for d in range(depth):
            for i in range(d_output.shape[1]):
                for j in range(d_output.shape[2]):
                    region = self.input[d, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size]
                    max_val = np.max(region)
                    mask = (region == max_val)
                    d_input[d, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size] += mask * d_output[d, i, j]
        
        return d_input

# Fully Connected Layer
class FCLayer:
    def __init__(self, input_size, output_size, learning_rate):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros(output_size)
        self.learning_rate = learning_rate

    def forward(self, input_vector):
        self.input = input_vector
        return np.dot(input_vector, self.weights) + self.biases

    def backward(self, d_output):
        d_weights = np.outer(self.input, d_output)
        d_input = np.dot(d_output, self.weights.T)
        
        self.weights -= self.learning_rate * d_weights
        self.biases -= self.learning_rate * d_output
        
        return d_input

# Utility Functions
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def cross_entropy_loss(predictions, label):
    return -np.log(predictions[label] + 1e-9)

def d_softmax_cross_entropy(predictions, label):
    grad = predictions.copy()
    grad[label] -= 1
    return grad

def flatten(volume):
    return volume.flatten()

def unflatten(vector, depth, height, width):
    return vector.reshape((depth, height, width))

# Load Dataset
print("Loading MNIST dataset...")
train_images = load_mnist_images("train-images.idx3-ubyte")
train_labels = load_labels("train-labels.idx1-ubyte")
print(f"Loaded {len(train_images)} training images.")

# Initialize Layers
conv_lr = 0.001
fc_lr = 0.001
num_filters = 8
filter_size = 3
conv = ConvLayer(num_filters, filter_size, conv_lr)
pool = MaxPoolLayer(2)
fc_input_size = num_filters * 13 * 13  # Output from pooling layer
num_classes = 10
fc = FCLayer(fc_input_size, num_classes, fc_lr)

# Training
epochs = 3
num_samples = len(train_images)

print("Starting training...")
start_time = time.time()

for epoch in range(epochs):
    total_loss = 0.0
    correct = 0

    for idx in range(num_samples):
        image = train_images[idx]
        label = train_labels[idx]

        # Forward pass
        conv_out = conv.forward(image)
        pool_out = pool.forward(conv_out)
        flat = flatten(pool_out)
        fc_out = fc.forward(flat)
        probs = softmax(fc_out)
        
        # Compute loss
        loss = cross_entropy_loss(probs, label)
        total_loss += loss
        
        # Accuracy calculation
        predicted = np.argmax(probs)
        if predicted == label:
            correct += 1

        # Backward pass
        d_loss = d_softmax_cross_entropy(probs, label)
        d_fc = fc.backward(d_loss)
        d_pool = unflatten(d_fc, num_filters, 13, 13)
        d_conv = pool.backward(d_pool)
        conv.backward(d_conv)

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} samples.")

    avg_loss = total_loss / num_samples
    accuracy = (correct / num_samples) * 100
    print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

# Print Final Accuracy
final_accuracy = (correct / num_samples) * 100
print(f"Final Training Accuracy: {final_accuracy:.2f}%")

end_time = time.time()
print(f"Total training time: {end_time - start_time:.2f} seconds")
