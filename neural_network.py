"""
Assignment 3: Unsupervised Learning and Dimensionality Reduction
Created by: Jon-Erik Akashi (jakashi3@gatech.edu)
Date: 3/10/2023

Notes:
    1. Silhouette plotting was taken from the https://scikit-learn.org/stable/auto_examples/cluster/
        Sample code was referenced here
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.cluster as skc
import sklearn.ensemble as ske
import sklearn.preprocessing as skp
import sklearn.model_selection as skm
import sklearn.ensemble as sken
import numpy as np
import sklearn.mixture as skmix
import sklearn.decomposition as skd
import sklearn.random_projection as skrp


def process_data(dataset):
    df = pd.read_csv(f"datasets/{dataset}.csv")

    if dataset == "titanic":
        pred_col = "Survived"

    elif dataset == "winequality-red":
        pred_col = "quality"

    else:
        df["Class"] = df["Class"].replace(["SEKER", "BARBUNYA", "BOMBAY", "CALI", "HOROZ", "SIRA", "DERMASON"],
                                          [0, 1, 2, 3, 4, 5, 6])
        pred_col = "Class"

    X = df.drop(pred_col, axis=1)
    y = df[pred_col]

    mm_scaler = MinMaxScaler()
    X = pd.DataFrame(mm_scaler.fit_transform(X.values))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=909)
    return df, X_train, X_test, y_train, y_test


def RandomForest(df, X, y):
    mm_scaler = skm.preprocessing.MinMaxScaler()
    adjusted_x = pd.DataFrame(mm_scaler.fit_transform(df.values))

    tree = sken.RandomForestClassifier(n_estimators=100, random_state=909)
    tree.fit(X, y)
    importance = tree.feature_importances_
    indices = np.argsort(importance)[::-1][:11]
    rand_x = adjusted_x.drop(columns=[col for col in adjusted_x if col not in indices])
    return rand_x


def choose_method(method, df, X, y):
    df, y = None, None
    if method == "k_means":
        df = df.join(pd.DataFrame(skc.KMeans(n_clusters=5, random_state=909).fit(df).predict(df), columns=["Clusters"]))
    elif method == "expectation_maximization":
        df = df.join(
            pd.DataFrame(skmix.GaussianMixture(n_components=5, random_state=909).fit(df).predict(df), columns=["Clusters"]))
    elif method == "pca":
        df = pd.DataFrame(skd.PCA(8).fit_transform(df))
    elif method == "ica":
        df = pd.DataFrame(skd.FastICA(7, random_state=909).fit_transform(df))
    elif method == "rca":
        df = pd.DataFrame(skrp.GaussianRandomProjection(n_components=13, random_state=909).fit_transform(df))
    elif method == "random_forest":
        df, y = RandomForest(df, X, y)
    elif method == "normal":
        df, y = df, y

    foo = df.values
    scaler = skm.preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(foo)
    x_scaled = pd.DataFrame(x_scaled)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=.2, shuffle=True, random_state=909)
    one_hot = skp.OneHotEncoder()
    y_train = one_hot.fit_transform(y_train.values.reshape(-1, 1)).todense()
    y_test = one_hot.transform(y_test.values.reshape(-1, 1)).todense()
    return x_train, x_test, y_train, y_test


def main():
    datasets = ["bean"]

    for dataset in datasets:
        # pre-process
        df, X_train, X_test, y_train, y_test = process_data(dataset=dataset)

        methods = ["k_means", "expectation_maximization", "pca", "ica", "rca", "random_forest", "normal"]
        for method in methods:
            new_X_train, new_X_test, new_y_train, new_y_test = choose_method(method=method, df=df, X=X_train, y=y_train)






if __name__ == "__main__":
    main()
