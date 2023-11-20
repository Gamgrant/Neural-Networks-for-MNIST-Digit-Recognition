"""Main script for the solution."""

import numpy as np
import pandas as pd
import argparse
import os
import npnn


def _get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", help="learning rate", type=float, default=0.1)
    p.add_argument("--opt", help="optimizer", default="SGD")
    p.add_argument(
        "--epochs", help="number of epochs to train", type=int, default=20)
    p.add_argument(
        "--save_stats", help="Save statistics to file", action="store_true")
    p.add_argument(
        "--save_pred", help="Save predictions to file", action="store_true")
    p.add_argument("--dataset", help="Dataset file", default="mnist.npz")
    p.add_argument(
        "--test_dataset", help="Dataset file (test set)",
        default="mnist_test.npz")
    p.set_defaults(save_stats=False, save_pred=False)
    return p.parse_args()


if __name__ == '__main__':
    args = _get_args()
    save_dir = "data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    X, y = npnn.load_mnist(args.dataset)
    indices = np.random.permutation(X.shape[0])
    train_i, val_i = indices[1000:], indices[:1000]

    X_train, y_train = X[train_i], y[train_i]
    X_val, y_val = X[val_i], y[val_i]

    train_dataset = npnn.Dataset(X_train, y_train, batch_size=32)
    val_dataset = npnn.Dataset(X_val, y_val, batch_size=32)

    modules = [
        npnn.Flatten(),
        npnn.Dense(28*28, 256),
        npnn.ELU(0.9),
        npnn.Dense(256, 64),
        npnn.ELU(0.9),
        npnn.Dense(64, 10),   
    ]

    optimizer = npnn.SGD(args.lr) if args.opt == "SGD" else npnn.Adam(learning_rate=0.002)
    model = npnn.Sequential(modules=modules, loss=npnn.SoftmaxCrossEntropy(), optimizer=optimizer)
    stats_data = []

    for e in range(args.epochs):
        print(f"Epoch {e}/{args.epochs}")

        loss_train, acc_train = model.train(train_dataset)
        val_loss, acc_val = model.test(val_dataset)
        stats_data.append({
            'training_loss': loss_train,
            'train_accuracies': acc_train,
            'val_losses': val_loss,
            'val_accuracies': acc_val
        })

    stats = pd.DataFrame(stats_data, columns=['training_loss', 'train_accuracies', 'val_losses', 'val_accuracies'])
    print(f"End of epoch {e}, Training Loss: {loss_train}, Accuracy: {acc_train}")
    print(f"Validation Loss: {val_loss}, Accuracy: {acc_val}")
    print("\nFinal Results:")
    print(stats.tail(1))

    # Save statistics to file.
    # We recommend that you save your results to a file, then plot them
    # separately, though you can also place your plotting code here.
    if args.save_stats:
        stats.to_csv(f"{save_dir}/{args.opt}_{args.lr}.csv")

    # Save predictions.
    if args.save_pred:
        X_test, _ = npnn.load_mnist("mnist_test.npz")
        y_pred = np.argmax(model.forward(X_test), axis=1).astype(np.uint8)
        np.save("mnist_test_pred.npy", y_pred)
