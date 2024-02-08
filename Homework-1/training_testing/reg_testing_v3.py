import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def plot_decision_boundary(model, X_test, y_test, test_pred, dataset):
    """
    Plots the decision boundary of a classification model along with the test data.

    Parameters:
    model: Trained classification model which has a predict method.
    X_test: Test data features (numpy array).
    y_test: True labels for the test data (numpy array).
    test_pred: Predicted labels for the test data (numpy array).
    dataset: Name of the dataset (string) for the plot title.

    The function creates a plot of the model's decision boundary and overlays the test data points.
    Points are colored based on their true labels, and incorrectly classified points are circled.
    """
    # Create a mesh grid to cover the feature space
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))

    # Predict classes for each point in the mesh grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_tensor = torch.from_numpy(grid_points).float()
    with torch.no_grad():
        grid_predictions_tensor = model.predict(grid_points_tensor)
        grid_predictions = grid_predictions_tensor.cpu().numpy()
        grid_predictions = grid_predictions.reshape(xx.shape)

    # Prepare the test data for plotting
    df_test = pd.DataFrame(X_test, columns=["Feature 1", "Feature 2"])
    df_test["Label"] = y_test.squeeze()
    df_test["Label"] = df_test["Label"].astype(int)

    incorrect_predictions = test_pred.cpu().numpy() != y_test

    plt.figure(figsize=(9, 5), dpi=120)
    plt.contourf(xx, yy, grid_predictions, alpha=0.2, cmap="RdBu_r")
    sns.scatterplot(
        data=df_test, x="Feature 1", y="Feature 2", hue="Label", palette=["blue", "red"]
    )

    # Highlight incorrect predictions
    incorrect_points = df_test[incorrect_predictions]
    plt.scatter(
        incorrect_points["Feature 1"],
        incorrect_points["Feature 2"],
        facecolors="none",
        edgecolors="black",
        s=100,
        label="Incorrectly Classified",
    )

    plt.title(f"{dataset} decision boundaries (mean-squared error)")
    plt.legend(title="Legend", loc="upper right")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.savefig(f"results/{dataset}/reg_l2_decision_boundary.png")
    plt.show()
    plt.close()
