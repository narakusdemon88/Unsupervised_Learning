from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from time import perf_counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import pandas as pd


def transform_data(df, method):
    df = df.dropna()
    if method == "kmeans":
        X = df.drop("Survived", axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=2)
        kmeans.fit(X_scaled)

        final_df = pd.DataFrame(X_scaled)
        final_df["Survived"] = kmeans.predict(X_scaled)

    elif method == "regular":
        final_df = df

    elif method == "expectation":
        X = df.drop("Survived", axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        em = GaussianMixture(n_components=2, random_state=42)
        em.fit(X_scaled)

        final_df = pd.DataFrame(X_scaled)
        final_df["Survived"] = em.predict(X_scaled)

    return final_df


def process_data(method):
    df = pd.read_csv("datasets/titanic.csv")

    # drop unnecessary stuff
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    # fix the non number columns
    df["Embarked"] = df["Embarked"].map(arg={"S": 0, "C": 1, "Q": 2})
    df["Sex"] = df["Sex"].map(arg={"male": 0, "female": 1})

    df = df.dropna()

    df = transform_data(df, method)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=909)
    return X_train, X_test, y_train, y_test


def plot_learning_curve(method):
    x_train, x_test, y_train, y_test = process_data(method)

    x_train_len = len(x_train)

    size_division = x_train_len * 0.1

    sizes = np.linspace(size_division, x_train_len, 10, dtype=int)[0:-1]

    size_per = [
        x / np.linspace(size_division, x_train_len, 10, dtype=int)[0:-1][-1] for x in
        np.linspace(size_division, x_train_len, 10, dtype=int)[0:-1]
    ]

    gd = MLPClassifier(max_iter=500)
    alpha = np.logspace(-1, 2, 5)
    learning_rate = np.logspace(-5, 0, 6)
    hidden_layer = [[i] for i in range(3, 5, 1)]

    params = {'alpha': alpha, 'learning_rate_init': learning_rate, 'hidden_layer_sizes': hidden_layer}
    gd = GridSearchCV(gd, param_grid=params, cv=10, n_jobs=-1)
    gd.fit(np.asarray(x_train), np.asarray(y_train))
    clf = gd.best_estimator_

    train_sizes, train_scores, cv_scores = learning_curve(clf, x_train,
                                                          y_train, train_sizes=sizes, cv=10,
                                                          scoring="f1_weighted")

    train_scores_mean = train_scores.mean(axis=1)
    cv_scores_mean = cv_scores.mean(axis=1)
    plt.plot(size_per, train_scores_mean, 'o-', color='r', label='Training Score')
    plt.plot(size_per, cv_scores_mean, 'o-', color='b', label='Cross-Validation Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('Sample Size (%)')

    plt.title(f"Titanic {method} NN Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    # plt.savefig(f"{self.data} LC - {method}")
    plt.show()
    plt.clf()


def plot_other_graphs(method):
    X_train, X_test, y_train, y_test = process_data(method)

    test_f1 = []
    times = []

    gd = MLPClassifier(max_iter=1000, random_state=909)

    params = {"alpha": np.logspace(-1, 2, 5),
              "learning_rate_init": np.logspace(-5, 0, 6),
              "hidden_layer_sizes": [[i] for i in range(1, 5)]}

    gd = GridSearchCV(gd, param_grid=params, cv=10, n_jobs=-1)

    gd.fit(X_train, y_train)
    gd = gd.best_estimator_
    # gd_loss = gd.loss_curve_

    for iteration in range(10, 101, 10):
        print(iteration)

        t1 = perf_counter()
        gd.set_params(max_iter=iteration)
        gd.fit(X_train, y_train)
        t2 = perf_counter()

        test_f1.append(f1_score(y_test, gd.predict(X_test), average="weighted"))
        times.append(t2 - t1)
        # gd_loss.append(gd.loss)

    print(f"F1 Scores:\n{test_f1}")
    print(f"Times:\n{times}")
    print(f"Loss:\n{gd.loss_curve_[::len(gd.loss_curve_) // 10][:10]}")

    # PLOT THE F1 GRAPH
    plt.plot(range(10, 101, 10), test_f1, label="Regular")
    plt.legend()
    plt.title("F1 Scores for Clustering Types")
    plt.xlabel("Iterations")
    plt.ylabel("F1 score")
    plt.show()
    plt.clf()

    # PLOT THE TIMES GRAPH
    plt.plot(range(10, 101, 10), times, label="Regular")
    plt.legend()
    plt.title("Prediction Times")
    plt.xlabel("Iterations")
    plt.ylabel("Time (Seconds)")
    plt.show()
    plt.clf()

    # PLOT THE LOSS CURVE
    plt.plot(range(10, 101, 10), gd.loss_curve_[::len(gd.loss_curve_) // 10][:10], label="Regular")
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
    plt.clf()


def main():
    # Plot Learning Curves
    plot_learning_curve("regular")
    plot_learning_curve("kmeans")
    plot_learning_curve("expectation")

    # Plot Loss/Run/F1
    plot_other_graphs("regular")
    plot_other_graphs("kmeans")
    plot_other_graphs("expectation")


if __name__ == "__main__":
    main()
