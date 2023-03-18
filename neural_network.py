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


def main():
    datasets = ["bean"]

    for dataset in datasets:
        # pre-process
        df, X_train, X_test, y_train, y_test = process_data(dataset=dataset)


if __name__ == "__main__":
    main()
