'''This code is for Training face recognition
 author : Hyunah Oh
 data : 2020.01.22
 flow : Detection -> Alignment -> Normalization -> Embedding(load pretrained) -> Training Classification
'''

import numpy as np
from sys import argv
import mlflow
import mlflow.sklearn
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

from data import load_metadata

def main():
    alpha = float(argv[1]) if len(argv) > 1 else 0
    l1_ratio = float(argv[2]) if len(argv) > 2 else 0
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)

    metadata = load_metadata('images')
    targets = np.array([m.name for m in metadata])
    embedded = joblib.load('models/embedded_images.pkl')

    encoder = LabelEncoder()
    encoder.fit(targets)

    # Numerical encoding of identities
    y = encoder.transform(targets)

    train_idx = np.arange(metadata.shape[0]) % 2 != 0
    test_idx = np.arange(metadata.shape[0]) % 2 == 0

    # 50 train examples of 10 identities (5 examples each)
    X_train = embedded[train_idx]
    # 50 test examples of 10 identities (5 examples each)
    X_test = embedded[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    svc = LinearSVC()
    ela_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)

    #### Training classification ####
    knn.fit(X_train, y_train)
    svc.fit(X_train, y_train)
    ela_net.fit(X_train, y_train)

    acc_knn = accuracy_score(y_test, knn.predict(X_test))
    acc_svc = accuracy_score(y_test, svc.predict(X_test))
    # acc_ela = accuracy_score(y_test, ela_net.predict(X_test))

    # (mae, rmse, r2) = eval_metrics(y_test, ela_net.predict(X_test))
    # mlflow.log_metric("mae", mae)
    # mlflow.log_metric("rmse", rmse)
    # mlflow.log_metric("r2", r2)

    # mlflow.log_artifact("train.parquet")
    # mlflow.log_artifact("plot.png")

    mlflow.sklearn.log_model(ela_net, "model")

    print(f'KNN accuracy = {acc_knn}, SVM accuracy = {acc_svc}')

    joblib.dump(svc, 'models/classification_model.pkl')

if __name__=="__main__":
    main()