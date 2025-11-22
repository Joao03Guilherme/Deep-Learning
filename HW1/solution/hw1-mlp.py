#!/usr/bin/env python

# Deep Learning Homework 1 - Multi Layer Perceptron

import argparse
import time
import pickle
import json
import os

import numpy as np

import utils

class MLP:
    def __init__(self, n_classes, n_features, hidden_size, learning_rate=0.001):
        self.n_classes = n_classes
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.lr = learning_rate
        
        # Initialize weights and biases
        # wij ~ N(mu, sigma^2) with mu=0.1 and sigma^2=0.1^2
        mu = 0.1
        sigma = 0.1
        
        self.W1 = np.random.normal(mu, sigma, (hidden_size, n_features))
        self.b1 = np.zeros(hidden_size)
        
        self.W2 = np.random.normal(mu, sigma, (n_classes, hidden_size))
        self.b2 = np.zeros(n_classes)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def forward(self, x):
        # x: (n_features,)
        
        # Layer 1
        z1 = np.dot(self.W1, x) + self.b1
        h1 = np.maximum(0, z1) # ReLU
        
        # Layer 2
        z2 = np.dot(self.W2, h1) + self.b2
        
        # Softmax
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores)
        
        return probs, h1, z1

    def predict(self, X):
        y_hat = []
        for x in X:
            probs, _, _ = self.forward(x)
            y_hat.append(np.argmax(probs))
        return np.array(y_hat)
    
    def evaluate(self, X, y):
        y_hat = self.predict(X)
        accuracy = np.mean(y_hat == y)
        return accuracy

    def train_epoch(self, X, y):
        total_loss = 0
        n_samples = X.shape[0]
        
        # Shuffle
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        for i in indices:
            x = X[i]
            target = y[i]
            
            # Forward
            probs, h1, z1 = self.forward(x)
            
            # Loss (Cross Entropy)
            loss = -np.log(probs[target])
            total_loss += loss
            
            # Backward
            # dL/dz2 = p - y_onehot
            delta2 = probs.copy()
            delta2[target] -= 1
            
            # dL/dW2 = delta2 * h1^T
            grad_W2 = np.outer(delta2, h1)
            grad_b2 = delta2
            
            # dL/dh1 = W2^T * delta2
            delta1 = np.dot(self.W2.T, delta2)
            
            # ReLU derivative: 1 if z1 > 0 else 0
            delta1[z1 <= 0] = 0
            
            # dL/dW1 = delta1 * x^T
            grad_W1 = np.outer(delta1, x)
            grad_b1 = delta1
            
            # Update
            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1
            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2
            
        return total_loss / n_samples

def main(args):
    utils.configure_seed(seed=args.seed)

    # Create output directory
    results_dir = os.path.join("Results", "MLP_results")
    os.makedirs(results_dir, exist_ok=True)

    # Update paths to save inside the results directory
    save_path = os.path.join(results_dir, os.path.basename(args.save_path))
    accuracy_plot_path = os.path.join(results_dir, os.path.basename(args.accuracy_plot))
    loss_plot_path = os.path.join(results_dir, "Q2-mlp-loss.pdf")
    scores_path = os.path.join(results_dir, os.path.basename(args.scores))

    # Load data without bias (bias is handled in the model)
    data = utils.load_dataset(data_path=args.data_path, bias=False)
    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]
    
    n_classes = np.unique(y_train).size
    n_feats = X_train.shape[1]

    # Initialize model
    model = MLP(n_classes, n_feats, args.hidden_size, args.learning_rate)

    epochs = np.arange(1, args.epochs + 1)
    
    train_losses = []
    train_accs = []
    valid_accs = []

    start = time.time()

    best_valid = 0.0
    best_epoch = -1
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        
        avg_loss = model.train_epoch(X_train, y_train)
        train_losses.append(avg_loss)
        
        train_acc = model.evaluate(X_train, y_train)
        valid_acc = model.evaluate(X_valid, y_valid)
        
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        
        print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(avg_loss, train_acc, valid_acc))

        if valid_acc > best_valid:
            best_valid = valid_acc
            best_epoch = i
            print(f"New best validation accuracy: {best_valid:.4f} at epoch {best_epoch}, saving model.")
            model.save(save_path)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))

    print("Reloading best checkpoint")
    best_model = MLP.load(save_path)
    test_acc = best_model.evaluate(X_test, y_test)

    print('Best model test acc: {:.4f}'.format(test_acc))

    # Plot accuracies
    utils.plot(
        "Epoch", "Accuracy",
        {"train": (epochs, train_accs), "valid": (epochs, valid_accs)},
        filename=accuracy_plot_path
    )
    
    # Plot loss
    utils.plot(
        "Epoch", "Loss",
        {"train": (epochs, train_losses)},
        filename=loss_plot_path
    )

    with open(scores_path, "w") as f:
        json.dump(
            {"best_valid": float(best_valid),
             "selected_epoch": int(best_epoch),
             "test": float(test_acc),
             "time": elapsed_time},
            f,
            indent=4
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int,
                        help="""Number of epochs to train for.""")
    parser.add_argument('--hidden-size', default=100, type=int,
                        help="""Hidden size.""")
    parser.add_argument('--learning-rate', default=0.001, type=float,
                        help="""Learning rate.""")
    parser.add_argument('--data-path', type=str, default="emnist-letters.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", default="mlp.pkl")
    parser.add_argument("--accuracy-plot", default="Q2-mlp-accs.pdf")
    parser.add_argument("--scores", default="Q2-mlp-scores.json")
    args = parser.parse_args()
    main(args)
            