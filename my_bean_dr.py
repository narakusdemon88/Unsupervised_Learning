"""
Assignment 3: Unsupervised Learning and Dimensionality Reduction
Created by: Jon-Erik Akashi (jakashi3@gatech.edu)
Date: 3/10/2023

Notes:
    1. Silhouette plotting was taken from the https://scikit-learn.org/stable/auto_examples/cluster/
        Sample code was referenced here
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import yellowbrick.cluster as ybc
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from sklearn.mixture import GaussianMixture
from matplotlib import cm
import sklearn.decomposition as skd
import scipy.stats as scs
import sklearn.random_projection as srp
import sklearn.ensemble as ske


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


def calculate_homogeneous_k_means(X, y, dataset, dimension):
    cluster_range = [2, 3, 4, 5, 6, 7, 8, 9]
    homogenity_lst = []
    for i in cluster_range:
        m_labels = KMeans(i, random_state=909).fit(X).labels_
        homogenity_lst.append(skm.homogeneity_score(y, m_labels))

    # do the plot here
    plt.plot(cluster_range, homogenity_lst)

    plt.tight_layout()

    plt.ylabel("Homogenity Score")
    plt.xlabel("Cluster Sizes")

    plt.grid()

    plt.title(f"Homogeneity Score for k-Means for {dataset} ({dimension})")

    plt.show()
    plt.clf()


def calculate_homogeneous_expectation_max(X, y, dataset):
    cluster_range = [2, 3, 4, 5, 6, 7, 8, 9]
    homogenity_lst = []
    for i in cluster_range:
        m = GaussianMixture(
            n_components=i,
            random_state=909).fit(X)
        homogenity_lst.append(skm.homogeneity_score(y, m.predict(X)))

    # TODO: change this below
    plt.plot(cluster_range, homogenity_lst, label='Homogeneity Score')

    plt.title(f"Expectation Maximization Homogeneity ({dataset})")
    plt.ylabel("Homogeneity")
    plt.xlabel("Components #")
    plt.tight_layout()
    plt.legend()

    plt.grid()
    plt.show()
    plt.clf()


def k_means_clustering(elbow, silhouette, X, y, dataset, dimension):
    if silhouette == True:
        plot_silhouette_k_means(X, dimension, "DR k-Means Silhouette")
    if elbow == True:
        elbow_results = calculate_elbow(KMeans(), "", X, dimension, dataset)
    calculate_homogeneous_k_means(X=X, y=y, dataset=dataset, dimension=dimension)


def plot_silhouette_k_means(X, dimension, name):
    clusters = [2, 3, 4, 5, 6]
    for cluster in clusters:
        m = KMeans(cluster, random_state=909)
        vis = ybc.SilhouetteVisualizer(estimator=m, colors="yellowbrick")
        vis.fit(X)
        print(f"sil score for {cluster} is {vis.silhouette_score_}")  # TODO: delete this later
        vis.show(outpath=f"{dimension} {name} {cluster}")   # TODO: change this up
        plt.clf()



def calculate_elbow(model, name, X, dimension, dataset):
    k_elbow_visualizer = ybc.KElbowVisualizer(model, metric="distortion", k=10, timings=False)
    k_elbow_visualizer.fit(X)
    outpath = f"K Means Elbow {dataset.upper()} for {dimension}"
    k_elbow_visualizer.show(outpath=outpath)
    plt.clf()
    elbow_value = k_elbow_visualizer.elbow_value_
    return elbow_value


def plot_silhouette_expectation_maximization(dimensionality_reduction, X):
    # TODO: CHANGE ALL IN FUNCTION
    cluster_range = [2, 3, 4, 5, 6]

    # TODO: change -999 to something else
    average_silhouette = -999

    for n_clusters in cluster_range:
        fig, (ax1) = plt.subplots(1, 1)

        ax1.set_xlim([
            -0.1,
            1
        ])

        ax1.set_ylim([
            0,
            len(X) + (n_clusters + 1) * 10
        ])

        clusterer = GaussianMixture(
            n_components=n_clusters,
            random_state=909)
        cluster_labels = clusterer.fit_predict(X)

        silhouette_avg = skm.silhouette_score(X, cluster_labels)
        print(
            "For n_components =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        if silhouette_avg > average_silhouette:
            average_silhouette = silhouette_avg

        sample_silhouette_values = skm.silhouette_samples(X, cluster_labels)

        lower_y = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = lower_y + size_cluster_i

            cmap = cm.get_cmap("Spectral")
            color = cmap(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(lower_y, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            ax1.text(-0.05, lower_y + 0.5 * size_cluster_i, str(i))
            lower_y = y_upper + 10

        ax1.set_title(f"The silhouette plot for EM {n_clusters} components.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()
    plt.clf()


def expectation_maximization(X):
    plot_silhouette_expectation_maximization(False, X)


def principal_components_analysis(X, dataset, df):
    # return pd.DataFrame(skd.PCA(n_components=8).fit_transform(X=df))
    return pd.DataFrame(skd.PCA(n_components=8).fit_transform(X=X))


def independent_components_analysis(X, dataset, df):
    return pd.DataFrame(skd.FastICA(n_components=7, random_state=909).fit_transform(X=X))


def random_components_analysis(X, dataset, df):
    return pd.DataFrame(srp.GaussianRandomProjection(n_components=13, random_state=909).fit_transform(X=X))



def random_forest(X, y, X_small):  # TODO: CHANGE VARIABLE NAMES AND OTHER STUFF
    tree = ske.RandomForestClassifier(n_estimators=100, random_state=909)
    tree.fit(X, y)
    indices = np.argsort(tree.feature_importances_)[::-1][:11]
    rand_x = X.drop(columns=[col for col in X if col not in indices])
    return rand_x, y


def main():
    datasets = ["bean"]

    for dataset in datasets:
        # pre-process
        df, X_train, X_test, y_train, y_test, X_small = process_data(dataset=dataset)

        # start Principal/Indepndent/Random Component Analysis
        # PCA_X = principal_components_analysis(X=X_train, dataset=dataset, df=df)
        # ICA_X = independent_components_analysis(X=X_train, dataset=dataset, df=df)
        # RCA_X = random_components_analysis(X=X_train, dataset=dataset, df=df)
        forest_x, forest_y = random_forest(X=X_train, y=y_train, X_small=X_small)

        # start K Means
        # k_means_clustering(True, True, PCA_X, y_train, dataset, dimension="PCA")
        # k_means_clustering(True, True, ICA_X, y_train, dataset, dimension="ICA")
        # k_means_clustering(True, True, RCA_X, y_train, dataset, dimension="RCA")
        k_means_clustering(True, True, forest_x, forest_y, dataset, dimension="randomforest")

        # start Expectation Maximization w/ Dimensionality Reduction
        expectation_maximization(X_train)




if __name__ == "__main__":
    main()
