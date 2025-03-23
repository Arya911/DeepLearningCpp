import tensorflow as tf
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


(x_train, y_train), (_, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)  # Reshape to (samples, 28, 28, 1)

# Build CNN model
model = Sequential([
    Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model and measure time
start_time = time.time()
history = model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=1)
end_time = time.time()

# Print training time
print(f"Total training time: {end_time - start_time:.2f} seconds")
