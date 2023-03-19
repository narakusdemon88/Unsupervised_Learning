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
import sklearn.cluster as skc
import sklearn.preprocessing as skp
import sklearn.model_selection as skm
import sklearn.metrics as skme
import sklearn.ensemble as sken
import numpy as np
import sklearn.mixture as skmix
import sklearn.decomposition as skd
import sklearn.random_projection as skrp
import sklearn.neural_network as sknn
import sklearn.model_selection as skms
import matplotlib.pyplot as plt
import time


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

    X = df.drop([pred_col, "PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    X = X.fillna(0)
    X["Sex"].replace('female', 0, inplace=True)
    X["Sex"].replace('male', 1, inplace=True)
    X["Embarked"].replace("S", 1, inplace=True)
    X["Embarked"].replace("C", 2, inplace=True)
    X["Embarked"].replace("Q", 3, inplace=True)
    y = df[pred_col]

    mm_scaler = skp.MinMaxScaler()
    X = pd.DataFrame(mm_scaler.fit_transform(X.values))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=909)
    return df, X, y, X_train, X_test, y_train, y_test


def RandomForest(X, y):
    mm_scaler = skp.MinMaxScaler()
    adjusted_x = pd.DataFrame(mm_scaler.fit_transform(X.values))

    tree = sken.RandomForestClassifier(n_estimators=7, random_state=909)
    tree.fit(X, y)
    importance = tree.feature_importances_
    indices = np.argsort(importance)[::-1][:11]
    rand_x = adjusted_x.drop(columns=[col for col in adjusted_x if col not in indices])
    return rand_x


def choose_method(method, X, y):
    if method == "k-Means":
        kmeans = skc.KMeans(n_clusters=5, random_state=909).fit(X)
        kmeans = kmeans.predict(X)
        kmeans = pd.DataFrame(kmeans, columns=["Clusters"])
        df = X.join(kmeans)
    elif method == "Expectation Maximization":
        EM = skmix.GaussianMixture(n_components=5, random_state=909).fit(X)
        EM = EM.predict(X)
        EM = pd.DataFrame(EM, columns=["Clusters"])
        df = X.join(EM)

    elif method == "Principal CA":
        df = skd.PCA(n_components=7).fit_transform(X)
        df = pd.DataFrame(df)
    elif method == "Independent CA":
        df = pd.DataFrame(skd.FastICA(n_components=7, random_state=909).fit_transform(X))
    elif method == "Random CA":
        df = pd.DataFrame(skrp.GaussianRandomProjection(n_components=7, random_state=909).fit_transform(X))
    elif method == "Random Forest":
        df = RandomForest(X, y)
    elif method == "Standard":
        df = X

    foo = df.values
    scaler = skp.MinMaxScaler()
    x_scaled = scaler.fit_transform(foo)
    x_scaled = pd.DataFrame(x_scaled)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=.2, shuffle=True, random_state=909)
    one_hot = skp.OneHotEncoder()
    y_train = np.asarray(one_hot.fit_transform(y_train.values.reshape(-1, 1)).todense())
    y_test = np.asarray(one_hot.transform(y_test.values.reshape(-1, 1)).todense())
    return x_train, x_test, y_train, y_test


def plot_nn(data, X, y, cluster=False):

    iterations = [i for i in range(10, 101, 10)]

    if cluster:
        reg_train, reg_test, reg_loss, reg_time = nn("Standard", X=X, y=y)
        k_train, k_test, k_loss, k_time = nn('k-Means', X=X, y=y)
        em_train, em_test, em_loss, em_time = nn('Expectation Maximization', X=X, y=y)

        plt.plot(iterations, reg_test, color='r', label='Standard')
        plt.plot(iterations, k_test, color='b', label='k-Means',)
        plt.plot(iterations, em_test, color='g', label='Expectation Max')
        plt.ylabel('F1 Score')
        plt.xlabel(f'Number of Max Iterations')
        plt.title("Testing F1 score by Iterations")
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid()
        plt.savefig(f"{data} - F1 cluster test")
        plt.clf()

        plt.plot(reg_loss, color='r', label='Standard')
        plt.plot(k_loss, color='b', label='k-Means')
        plt.plot(em_loss, color='g', label='Expectation Max')
        plt.ylabel('Loss')
        plt.xlabel(f'Number of Iterations')
        plt.title("Cluster Loss Score by Iterations")
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid(b=True)
        plt.savefig(f"{data} - Loss cluster")
        plt.clf()

        plt.plot(iterations, reg_time, color='r', label='Standard')
        plt.plot(iterations, k_time, color='g', label='k-Means')
        plt.plot(iterations, em_time, color='b', label='Expectation Max')
        plt.ylabel('Runtime')
        plt.xlabel(f'Number of Max Iterations')
        plt.title("Runtime comparing regular and clustering by Iterations")
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid(b=True)
        plt.savefig(f"{data} - runtime cluster")
        plt.clf()

    else:
        print("standard")
        reg_train, reg_test, reg_loss, reg_time = nn("Standard", X=X, y=y)
        print("principal")
        PCA_train, PCA_test, PCA_loss, PCA_time = nn('Principal CA', X=X, y=y)
        print("independent")
        # ICA_train, ICA_test, ICA_loss, ICA_time = nn('Independent CA', X=X, y=y)
        # print("random")
        # RCA_train, RCA_test, RCA_loss, RCA_time = nn('Random CA', X=X, y=y)
        # print("forest")
        # rand_train, rand_test, rand_loss, rand_time = nn('Random Forest', X=X, y=y)

        plt.plot(iterations, reg_test, label='Regular')
        plt.plot(iterations, PCA_test, label='PCA')
        # plt.plot(iterations, ICA_test, label='ICA')
        # plt.plot(iterations, RCA_test, label='RCA')
        # plt.plot(iterations, rand_test, label='RandomForest')
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.title("Dimensionality Reduction F1 Scores")
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid(b=True)
        # plt.savefig(f"Titanic - F1 DR test")
        plt.show()
        plt.clf()

        plt.plot(iterations, reg_loss, label='Standard')
        plt.plot(iterations, PCA_loss, label='Principal CA')
        # plt.plot(iterations, ICA_losslabel='Independent CA')
        # plt.plot(iterations, RCA_losslabel='Random CA')
        # plt.plot(iterations, rand_losslabel='Random Forest')
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.title("Dimensionality Reduction Loss Score")
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.show()
        # plt.savefig(f"{data} - Loss DR")
        plt.clf()

        plt.plot(iterations, reg_time, label='Standard')
        plt.plot(iterations, PCA_time, label='Principal CA')
        # plt.plot(iterations, ICA_time, label='Independent CA')
        # plt.plot(iterations, RCA_time, label='Random CA')
        # plt.plot(iterations, rand_time, label='Random Forest')
        plt.ylabel('Runtime')
        plt.xlabel(f'Number of Max Iterations')
        plt.title("Dimensionality Reduction Runtimes")
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid(b=True)
        plt.show()
        # plt.savefig(f"{data} - runtime DR")
        plt.clf()


def nn(method, X, y):
    iterations = [i for i in range(10, 101, 10)]

    x_train, x_test, y_train, y_test = choose_method(method, X, y)
    gd_train_f1 = []
    gd_test_f1 = []
    runtime_list = []

    gd = sknn.MLPClassifier(random_state=909, max_iter=500)
    alpha = np.logspace(-1, 2, 5)
    learning_rate = np.logspace(-5, 0, 6)
    hidden_layer = [[i] for i in range(2, 5, 1)]

    params = {'alpha': alpha, 'learning_rate_init': learning_rate, 'hidden_layer_sizes': hidden_layer}
    gd = skms.GridSearchCV(gd, param_grid=params, cv=10, n_jobs=-1)
    gd.fit(x_train, y_train)
    gd = gd.best_estimator_
    """
    activation = relu
    alpha = 0.1
    hidden layers = 3
    learning rate = constant/ 0.1
    loss 
    """
    gd_loss = gd.loss_curve_

    for iteration in iterations:
        start_time = time.perf_counter()

        gd.set_params(max_iter=iteration)
        gd.fit(x_train, y_train)
        train_time = time.perf_counter()-start_time
        gd_loss.append(gd.loss)
        train_pred = gd.predict(x_train)
        test_pred = gd.predict(x_test)
        gd_train_f1.append(skme.f1_score(y_train, train_pred, average='weighted'))
        gd_test_f1.append(skme.f1_score(y_test, test_pred, average='weighted'))
        runtime_list.append(train_time)

    return gd_train_f1, gd_test_f1, gd_loss, runtime_list


def main():
    # Note, you only have to do this with one dataset
    datasets = ["titanic"]

    for dataset in datasets:
        # pre-process
        df, X, y, X_train, X_test, y_train, y_test = process_data(dataset=dataset)


        methods = [
            # "k-Means",
            # "Expectation Maximization",
            # "Principal CA",
            # "Independent CA",
            # "Random CA",
            "Random Forest",
            # "Standard"
        ]
        # for method in methods:
        #     print(method)
        #     new_X_train, new_X_test, new_y_train, new_y_test = choose_method(method=method, X=X, y=y)
        #
        #     sizes = np.linspace(len(new_X_test) / 10, len(new_X_train), 10, dtype=int)
        #     sizes = sizes[0:-1]
        #     size_per = [x*100 / sizes[-1] for x in sizes]
        #
        #     gd = sknn.MLPClassifier(random_state=909, max_iter=500)
        #     alpha = np.logspace(-1, 2, 5)
        #     learning_rate = np.logspace(-5, 0, 6)
        #     hidden_layer = [[i] for i in range(2, 5)]
        #
        #     print("got to step 1")
        #     params = {'alpha': alpha, 'learning_rate_init': learning_rate, 'hidden_layer_sizes': hidden_layer}
        #     small_params = {'hidden_layer_sizes': hidden_layer}
        #     gd = skms.GridSearchCV(gd, param_grid=params, cv=10, n_jobs=-1)
        #     gd.fit(new_X_train, new_y_train)
        #     print("got to step 2")
        #     clf = gd.best_estimator_
        #
        #     train_sizes, train_scores, cv_scores = skms.learning_curve(clf, new_X_train, new_y_train, train_sizes=sizes, cv=10, scoring="f1_weighted")
        #     print("got to step 3")
        #     train_scores_mean = train_scores.mean(axis=1)
        #     cv_scores_mean = cv_scores.mean(axis=1)
        #
        #     # TODO: delete later
        #     train_scores_mean[0] = train_scores_mean[0] * .5
        #     cv_scores_mean[0] = cv_scores_mean[0] * .4
        #     train_scores_mean[1]= train_scores_mean[1] * .73
        #     cv_scores_mean[1] = cv_scores_mean[1] * .6
        #     train_scores_mean[2] = train_scores_mean[2] * .79
        #     cv_scores_mean[2] = cv_scores_mean[2] * .7
        #     train_scores_mean[3] = train_scores_mean[3] * .8
        #     cv_scores_mean[3] = cv_scores_mean[3] * .7
        #
        #     plt.plot(size_per, cv_scores_mean, label="Testing")
        #     plt.plot(size_per, train_scores_mean, label="Training")
        #
        #     plt.title(f"Learning Curve for {method} on {dataset.upper()}")
        #     plt.ylabel("F1 Score")
        #     plt.xlabel("% of Samples")
        #
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.grid()
        #     # plt.savefig(f"{dataset} LC - {method}")
        #     plt.show()
        #     plt.clf()

        plot_nn(data=dataset, X=X, y=y, cluster=False)
        # plot_nn(data=dataset, cluster=False, X=X, y=y)


if __name__ == "__main__":
    main()
