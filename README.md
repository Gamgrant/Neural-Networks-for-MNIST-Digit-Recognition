# Neural-Networks-for-MNIST-Digit-Recognition
Neural Networks for MNIST Digit Recognition
The classic MNIST dataset consists of 28x28 pixel (784 features) grayscale images of handwritten digits 0-9 (10 classes).

In this problem, I implemented a neural network to classify the MNIST dataset using raw pixels as features. Specifically, I implemented a type of architecture called a Multi-Layered Perceptron (MLP), which consists of several Dense layers (i.e. “Perceptrons”) connected sequentially, with nonlinear activation functions after each layer. Then I trained this network using the Categorical Cross-Entropy loss function, usign SGD and Adam optimizer

# Specs
![specs](specs/specs1.png)
![specs](specs/specs2.png)

# Training & Validation Loss & Accuracy vs Epochs

![Training & Validation Loss & Accuracy vs Epochs](results/results1.png)

# Lowest Validation Loss using SGD

![Lowest Validation Loss using SGD](results/lowest_validation_loss_SGD.png)

# Lowest Validation Loss using Adam

![Lowest Validation Loss using SGD](results/lowest_validation_loss_adam.png)



