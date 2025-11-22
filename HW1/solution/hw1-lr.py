#!/usr/bin/env python

# Deep Learning Homework 1 - Logistic Regression

import argparse
import time
import pickle
import json
import os
import copy

import numpy as np
from sklearn.decomposition import PCA

import utils

class LogisticRegression:
    def __init__(self, n_classes, n_features, regularization="l2"):
        self.W = np.zeros((n_classes, n_features))
        self.regularization = regularization

    def save(self, path):
        """
        Save logistic regression model to the provided path
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """
        Load logistic regression model from the provided path
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict_proba(self, X):
        """
        X (n_examples, n_features)
        returns predicted probabilities y_hat, whose shape is (n_examples, n_classes)
        """
        logits = X @ self.W.T 
        
        # Softmax
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probabilities
    
    def train_epoch(self, X, y, lr, l2_decay):
        """
        X (n_examples, n_features): features for the whole dataset
        y (n_examples,): labels for the whole dataset
        lr (float): learning rate
        l2_decay (float): L2 regularization strength
        """
        n_examples = X.shape[0]
        indices = np.arange(n_examples)
        np.random.shuffle(indices)
        
        for i in indices:
            x_i = X[i] # (n_features,)
            y_i = y[i] # scalar
            
            # Forward
            scores = self.W @ x_i # (n_classes,)
            
            # Softmax
            scores -= np.max(scores)
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores) # (n_classes,)
            
            # Gradient
            # dL/dz = p - y
            d_scores = probs.copy()
            d_scores[y_i] -= 1
            
            # dL/dW = d_scores * x_i^T
            grad_W = np.outer(d_scores, x_i)
            
            # Regularization
            if self.regularization == 'l2':
                grad_W += 2 * l2_decay * self.W
            
            # Update
            self.W -= lr * grad_W

    def predict(self, X):
        """
        X (n_examples, n_features)
        returns predicted labels y_hat, whose shape is (n_examples,)
        """
        probabilities = self.predict_proba(X)
        y_hat = np.argmax(probabilities, axis=1)
        return y_hat
        
    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        returns accuracy (float)
        """
        y_hat = self.predict(X)
        accuracy = np.mean(y_hat == y)
        return accuracy


def add_bias(X):
    """Add a bias column of ones to X."""
    return np.hstack((X, np.ones((X.shape[0], 1))))

def get_projections(X):
    """
    1. Horizontal and Vertical Projections
    Reshape to 28x28.
    Sum rows -> 28 features.
    Sum cols -> 28 features.
    Concatenate -> 56 features.
    """
    N = X.shape[0]
    # Reshape to (N, 28, 28)
    images = X.reshape(N, 28, 28)
    
    # Horizontal projection: sum pixel values across each row (axis 2) -> (N, 28)
    h_proj = np.sum(images, axis=2)
    
    # Vertical projection: sum pixel values down each column (axis 1) -> (N, 28)
    v_proj = np.sum(images, axis=1)
    
    # Concatenate
    features = np.hstack((h_proj, v_proj))
    return features

def get_downsampled(X, pool_size=2):
    """
    2. Downsampling (Average Pooling)
    Reshape to 28x28.
    Average pool with pool_size x pool_size blocks.
    Flatten.
    """
    N = X.shape[0]
    img_size = 28
    new_size = img_size // pool_size
    
    images = X.reshape(N, img_size, img_size)
    
    # Reshape to (N, new_size, pool_size, new_size, pool_size)
    # Then mean over the pool_size dimensions (axis 2 and 4)
    downsampled = images.reshape(N, new_size, pool_size, new_size, pool_size).mean(axis=(2, 4))
    
    # Flatten
    features = downsampled.reshape(N, -1)
    return features

def apply_pca(X_train, X_valid, X_test, n_components=50):
    """
    3. PCA
    Fit PCA on X_train.
    Transform X_train, X_valid, X_test.
    """
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_valid_pca = pca.transform(X_valid)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_valid_pca, X_test_pca


def get_data(args, feature_type):
    # Load data (without bias initially)
    data = utils.load_dataset(args.data_path, bias=False)
    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]
    
    # Transform
    if feature_type == 'pixel':
        pass
    elif feature_type == 'projections':
        X_train = get_projections(X_train)
        X_valid = get_projections(X_valid)
        X_test = get_projections(X_test)
    elif feature_type == 'downsample':
        X_train = get_downsampled(X_train, pool_size=args.downsample_size)
        X_valid = get_downsampled(X_valid, pool_size=args.downsample_size)
        X_test = get_downsampled(X_test, pool_size=args.downsample_size)
    elif feature_type == 'pca':
        X_train, X_valid, X_test = apply_pca(X_train, X_valid, X_test, n_components=args.pca_components)
        
    # Add bias
    X_train = add_bias(X_train)
    X_valid = add_bias(X_valid)
    X_test = add_bias(X_test)
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def train_and_eval(args, X_train, y_train, X_valid, y_valid, X_test, y_test, name, results_dir, lr, l2):
    n_classes = np.unique(y_train).size
    n_features = X_train.shape[1]
    
    # Initialize model
    model = LogisticRegression(n_classes, n_features, regularization="l2")
    
    epochs = np.arange(1, args.epochs + 1)
    valid_accs = []
    train_accs = []
    
    start = time.time()
    
    best_valid = -1.0
    best_epoch = -1
    best_model = None
    
    # Unique save path for this experiment
    save_path = os.path.join(results_dir, f"best_model_{name}.pkl")
    
    for i in epochs:
        model.train_epoch(X_train, y_train, lr, l2)
        
        train_acc = model.evaluate(X_train, y_train)
        valid_acc = model.evaluate(X_valid, y_valid)
        
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_epoch = i
            best_model = copy.deepcopy(model)
            
    elapsed_time = time.time() - start
    
    # Save best model and evaluate
    if best_model is not None:
        best_model.save(save_path)
        test_acc = best_model.evaluate(X_test, y_test)
    else:
        test_acc = 0.0
    
    return {
        "name": name,
        "dim": n_features,
        "time": elapsed_time,
        "best_valid": best_valid,
        "test_acc": test_acc,
        "valid_accs": valid_accs,
        "train_accs": train_accs
    }

def run_grid_search(args, results_dir):
    lrs = [0.01, 0.001, 0.0001]
    l2s = [0.001, 0.0001]
    feature_types = ['pixel', 'projections', 'downsample', 'pca']
    
    results = []
    
    print(f"{'Feature':<12} | {'LR':<8} | {'L2':<8} | {'Val Acc':<8}")
    print("-" * 45)
    
    best_overall_val = -1
    best_overall_config = None
    best_overall_test = -1
    
    for f_type in feature_types:
        X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(args, f_type)
        
        for lr in lrs:
            for l2 in l2s:
                config_name = f"{f_type}_lr{lr}_l2{l2}"
                res = train_and_eval(args, X_train, y_train, X_valid, y_valid, X_test, y_test, config_name, results_dir, lr, l2)
                
                print(f"{f_type:<12} | {lr:<8} | {l2:<8} | {res['best_valid']:<8.4f}")
                
                results.append({
                    "feature": f_type,
                    "lr": lr,
                    "l2": l2,
                    "val_acc": res['best_valid'],
                    "test_acc": res['test_acc']
                })
                
                if res['best_valid'] > best_overall_val:
                    best_overall_val = res['best_valid']
                    best_overall_config = res
                    best_overall_test = res['test_acc']
                    
    print("\nGrid Search Complete.")
    print(f"Best Configuration: Feature={best_overall_config['name']}, Val Acc={best_overall_val:.4f}, Test Acc={best_overall_test:.4f}")
    
    # Save results
    with open(os.path.join(results_dir, "grid_search_results.json"), "w") as f:
        json.dump(results, f, indent=4)

def main(args):
    utils.configure_seed(seed=args.seed)
    
    # Create output directory
    results_dir = os.path.join("Results", "LR_results")
    os.makedirs(results_dir, exist_ok=True)
    
    if args.grid_search:
        run_grid_search(args, results_dir)
        return
    
    if args.compare_all:
        configs = [
            ('Pixel', 'pixel'),
            ('Projections', 'projections'),
            ('Downsample', 'downsample'),
            ('PCA', 'pca')
        ]
        
        results = []
        curves = {}
        epochs = np.arange(1, args.epochs + 1)
        
        print(f"{'Name':<15} | {'Dim':<5} | {'Time (s)':<8} | {'Val Acc':<8} | {'Test Acc':<8}")
        print("-" * 65)
        
        for name, f_type in configs:
            X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(args, f_type)
            
            res = train_and_eval(args, X_train, y_train, X_valid, y_valid, X_test, y_test, name, results_dir, args.learning_rate, args.l2_decay)
            results.append(res)
            curves[name] = (epochs, res['valid_accs'])
            
            print(f"{res['name']:<15} | {res['dim']:<5} | {res['time']:<8.2f} | {res['best_valid']:<8.4f} | {res['test_acc']:<8.4f}")
            
        # Plot comparison
        utils.plot("Epoch", "Validation Accuracy", curves, filename=os.path.join(results_dir, "comparison_plot.pdf"))
        
        # Save comparison json
        with open(os.path.join(results_dir, "comparison_results.json"), "w") as f:
            # Convert numpy types to python types for json
            json_results = []
            for r in results:
                r_copy = r.copy()
                del r_copy['valid_accs']
                del r_copy['train_accs']
                json_results.append(r_copy)
            json.dump(json_results, f, indent=4)
            
        print(f"\nComparison complete. Results saved to {results_dir}")
        return

    # Update paths to save inside the results directory
    save_path = os.path.join(results_dir, os.path.basename(args.save_path))
    accuracy_plot_path = os.path.join(results_dir, os.path.basename(args.accuracy_plot))
    scores_path = os.path.join(results_dir, os.path.basename(args.scores))
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(args, args.feature_type)
    
    n_classes = np.unique(y_train).size
    n_features = X_train.shape[1]
    print(f"Number of features after transformation: {n_features}")
    
    # Initialize model
    model = LogisticRegression(n_classes, n_features, regularization="l2")
    
    epochs = np.arange(1, args.epochs + 1)
    valid_accs = []
    train_accs = []
    
    start = time.time()
    
    best_valid = 0.0
    best_epoch = -1
    
    print(f"Training Logistic Regression with lr={args.learning_rate}, l2={args.l2_decay}")
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        
        model.train_epoch(X_train, y_train, args.learning_rate, args.l2_decay)
        
        train_acc = model.evaluate(X_train, y_train)
        valid_acc = model.evaluate(X_valid, y_valid)
        
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        
        print('train acc: {:.4f} | val acc: {:.4f}'.format(train_acc, valid_acc))
        
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
    best_model = LogisticRegression.load(save_path)
    test_acc = best_model.evaluate(X_test, y_test)
    
    print('Best model test acc: {:.4f}'.format(test_acc))
    
    utils.plot(
        "Epoch", "Accuracy",
        {"train": (epochs, train_accs), "valid": (epochs, valid_accs)},
        filename=accuracy_plot_path
    )
    
    with open(scores_path, "w") as f:
        json.dump({
            "best_valid": float(best_valid),
            "selected_epoch": int(best_epoch),
            "test": float(test_acc),
            "time": elapsed_time,
            "args": vars(args)
        }, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int, help="Number of epochs to train for.")
    parser.add_argument('--data-path', type=str, default="emnist-letters.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--l2-decay", type=float, default=0.00001)
    parser.add_argument("--save-path", default="lr_model.pkl")
    parser.add_argument("--accuracy-plot", default="Q2-lr-accs.pdf")
    parser.add_argument("--scores", default="Q2-lr-scores.json")
    
    # Feature representation arguments
    parser.add_argument("--feature-type", choices=['pixel', 'projections', 'downsample', 'pca'], default='pixel',
                        help="Type of feature representation to use.")
    parser.add_argument("--pca-components", type=int, default=50, help="Number of components for PCA.")
    parser.add_argument("--downsample-size", type=int, default=2, help="Pool size for downsampling (e.g. 2 for 14x14).")
    parser.add_argument("--compare-all", action="store_true", help="Run all feature representations and compare them.")
    parser.add_argument("--grid-search", action="store_true", help="Run grid search over hyperparameters and feature representations.")
    
    args = parser.parse_args()
    main(args)