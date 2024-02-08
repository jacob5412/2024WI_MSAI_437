import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_decision_boundary(X_test, y_test, test_pred, dataset):
    # Prepare the test data for plotting
    df_test = pd.DataFrame(X_test[:, 0:2], columns=["Feature 1", "Feature 2"])
    df_test["Label"] = y_test.squeeze()
    df_test["Label"] = df_test["Label"].astype(int)

    incorrect_predictions = test_pred != y_test.squeeze()

    plt.figure(figsize=(9, 5), dpi=120)
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

    plt.title(f"{dataset} errors mean-squared error)")
    plt.legend(title="Legend", loc="upper right")

    plt.savefig(f"results/{dataset}/rbf_decision_boundary.png")
    plt.show()
    plt.close()
