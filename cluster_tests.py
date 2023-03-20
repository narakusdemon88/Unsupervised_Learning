from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
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

        em = GaussianMixture(n_components=3, random_state=42)
        em.fit(X_scaled)

        final_df = pd.DataFrame(X_scaled)
        final_df["Survived"] = em.predict(X_scaled)

    return final_df


def process_data(method):
    df = pd.read_csv("titanic.csv")

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


def main():
    # X_train, X_test, y_train, y_test = process_data("regular")
    # X_train, X_test, y_train, y_test = process_data("kmeans")
    X_train, X_test, y_train, y_test = process_data("expectation")

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

if __name__ == "__main__":
    main()
