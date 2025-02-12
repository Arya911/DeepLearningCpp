all:
	g++ -O2 -std=c++11 cnn_mnist.cpp -o cnn_mnist
	./cnn_mnist
