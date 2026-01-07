#!/usr/bin/env python

# Deep Learning Homework 1 - Logistic Regression

import argparse
import time
import pickle
import json
import os
import copy
import matplotlib.pyplot as plt

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

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
            
            # Regularization (don't regularize bias - last column)
            if self.regularization == 'l2':
                reg = 2 * l2_decay * self.W.copy()
                reg[:, -1] = 0  # Exclude bias column
                grad_W += reg
            
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
    # standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_valid_pca = pca.transform(X_valid)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_valid_pca, X_test_pca

def apply_umap(X_train, X_valid, X_test, n_components=50):
    """
    4. UMAP
    Fit UMAP on X_train.
    Transform X_train, X_valid, X_test.
    """
    import umap
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    X_train_umap = reducer.fit_transform(X_train)
    X_valid_umap = reducer.transform(X_valid)
    X_test_umap = reducer.transform(X_test)
    return X_train_umap, X_valid_umap, X_test_umap

def get_hog_features(X, pixels_per_cell=(7, 7), cells_per_block=(2, 2), orientations=9):
    """
    5. HOG (Histogram of Oriented Gradients)
    Extract HOG features from images.
    HOG captures edge directions and gradient magnitudes, which is effective
    for character recognition since letters are defined by their edges.
    
    References:
    - Dalal, N., & Triggs, B. (2005). "Histograms of oriented gradients for human detection." CVPR.
    """
    from skimage.feature import hog
    N = X.shape[0]
    images = X.reshape(N, 28, 28)
    
    features = []
    for img in images:
        feat = hog(img, orientations=orientations, 
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block, 
                   block_norm='L2-Hys')
        features.append(feat)
    
    return np.array(features)

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
    elif feature_type == 'umap':
        X_train, X_valid, X_test = apply_umap(X_train, X_valid, X_test, n_components=args.pca_components)
    elif feature_type == 'hog':
        X_train = get_hog_features(X_train)
        X_valid = get_hog_features(X_valid)
        X_test = get_hog_features(X_test)
        
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
    
    # Epoch 0
    train_accs = [model.evaluate(X_train, y_train)]
    valid_accs = [model.evaluate(X_valid, y_valid)]
    
    epochs = np.arange(1, args.epochs + 1)
    
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
    """
    Grid search as specified in Question 2(c):
    - 3 learning rate values
    - 2 L2 penalty values
    - 2 feature representations (pixel and HOG)
    Total: 3 x 2 x 2 = 12 configurations
    """
    # Grid search parameters (exactly as required by the exercise)
    lrs = [0.01, 0.001, 0.0001]  # 3 learning rates
    l2s = [0.001, 0.00001]       # 2 L2 penalties
    feature_types = ['pixel', 'hog']  # Original (pixel) + alternative (HOG)
    
    # Path for results
    results_file = os.path.join(results_dir, "grid_search_results.json")
    
    # Delete previous results if --rerun is specified
    if args.rerun and os.path.exists(results_file):
        os.remove(results_file)
        print(f"Deleted: {results_file}")
        print("Previous grid search results cleared. Starting fresh.\n")
    
    results = []
    
    print("=" * 70)
    print("Grid Search: 3 LRs x 2 L2s x 2 Features = 12 configurations")
    print("=" * 70)
    print(f"{'#':<3} | {'Feature':<8} | {'LR':<10} | {'L2':<10} | {'Val Acc':<10} | {'Test Acc':<10}")
    print("-" * 70)
    
    best_overall_val = -1
    best_overall_config = None
    config_num = 0
    
    # Preload data for both feature types
    data_cache = {}
    for f_type in feature_types:
        print(f"Loading {f_type} features...")
        data_cache[f_type] = get_data(args, f_type)
    print()
    
    for f_type in feature_types:
        X_train, y_train, X_valid, y_valid, X_test, y_test = data_cache[f_type]
        
        for lr in lrs:
            for l2 in l2s:
                config_num += 1
                config_name = f"{f_type}_lr{lr}_l2{l2}"
                
                res = train_and_eval(args, X_train, y_train, X_valid, y_valid, X_test, y_test, 
                                   config_name, results_dir, lr, l2)
                
                print(f"{config_num:<3} | {f_type:<8} | {lr:<10} | {l2:<10} | {res['best_valid']:<10.4f} | {res['test_acc']:<10.4f}")
                
                result_entry = {
                    "config_num": config_num,
                    "feature": f_type,
                    "lr": lr,
                    "l2": l2,
                    "val_acc": res['best_valid'],
                    "test_acc": res['test_acc'],
                    "time": res['time'],
                    "dim": res['dim']
                }
                
                results.append(result_entry)
                
                if res['best_valid'] > best_overall_val:
                    best_overall_val = res['best_valid']
                    best_overall_config = result_entry
    
    # Print summary
    print("\n" + "=" * 70)
    print("GRID SEARCH RESULTS SUMMARY")
    print("=" * 70)
    
    # Print validation accuracy for every configuration (as required)
    print("\nValidation accuracy of every configuration:")
    print("-" * 50)
    for r in results:
        print(f"  Config {r['config_num']:2d}: {r['feature']:<6} | lr={r['lr']:<8} | l2={r['l2']:<10} | val_acc={r['val_acc']:.4f}")
    
    # Print best configuration and its test accuracy (as required)
    print("\n" + "-" * 50)
    print("BEST CONFIGURATION:")
    print(f"  Feature:           {best_overall_config['feature']}")
    print(f"  Learning Rate:     {best_overall_config['lr']}")
    print(f"  L2 Penalty:        {best_overall_config['l2']}")
    print(f"  Validation Acc:    {best_overall_config['val_acc']:.4f}")
    print(f"  Test Accuracy:     {best_overall_config['test_acc']:.4f}")
    print("=" * 70)
    
    # Save results
    summary = {
        "description": "Grid search results for Question 2(c)",
        "grid_parameters": {
            "learning_rates": lrs,
            "l2_penalties": l2s,
            "feature_types": feature_types
        },
        "total_configs": 12,
        "all_results": results,
        "best_config": best_overall_config
    }
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nResults saved to {results_file}")

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
            ('PCA', 'pca'),
            ('UMAP', 'umap'),
            ('HOG', 'hog')
        ]
        
        results = []
        curves = {}
        epochs = np.arange(1, args.epochs + 1)
        plot_epochs = np.arange(0, args.epochs + 1)
        
        print(f"{'Name':<15} | {'Dim':<5} | {'Time (s)':<8} | {'Val Acc':<8} | {'Test Acc':<8}")
        print("-" * 65)
        
        for name, f_type in configs:
            X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(args, f_type)
            
            res = train_and_eval(args, X_train, y_train, X_valid, y_valid, X_test, y_test, name, results_dir, args.learning_rate, args.l2_decay)
            results.append(res)
            curves[name] = (plot_epochs, res['valid_accs'])
            
            print(f"{res['name']:<15} | {res['dim']:<5} | {res['time']:<8.2f} | {res['best_valid']:<8.4f} | {res['test_acc']:<8.4f}")
            
        # Plot comparison
        plt.xticks(np.arange(0, args.epochs + 1, 5))
        plt.xlim(0, args.epochs)
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
    
    # Epoch 0
    train_accs = [model.evaluate(X_train, y_train)]
    valid_accs = [model.evaluate(X_valid, y_valid)]
    
    epochs = np.arange(1, args.epochs + 1)
    
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
    
    plot_epochs = np.arange(0, args.epochs + 1)
    plt.xticks(np.arange(0, args.epochs + 1, 5))
    plt.xlim(0, args.epochs)
    utils.plot(
        "Epoch", "Accuracy",
        {"train": (plot_epochs, train_accs), "valid": (plot_epochs, valid_accs)},
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
    parser.add_argument("--feature-type", choices=['pixel', 'projections', 'downsample', 'pca', 'umap', 'hog'], default='pixel',
                        help="Type of feature representation to use.")
    parser.add_argument("--pca-components", type=int, default=50, help="Number of components for PCA.")
    parser.add_argument("--downsample-size", type=int, default=2, help="Pool size for downsampling (e.g. 2 for 14x14).")
    parser.add_argument("--compare-all", action="store_true", help="Run all feature representations and compare them.")
    parser.add_argument("--grid-search", action="store_true", help="Run grid search over hyperparameters and feature representations.")
    parser.add_argument("--rerun", action="store_true", help="Delete previous grid search results and run from scratch.")
    
    args = parser.parse_args()
    main(args)