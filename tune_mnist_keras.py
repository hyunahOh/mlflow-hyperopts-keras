import mlflow
from mlflow.tracking import MlflowClient
import time
import random
import argparse
import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf

from ray.tune.integration.keras import TuneReporterCallback
from ray.tune.logger import MLFLowLogger, DEFAULT_LOGGERS

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing")
args, _ = parser.parse_known_args()


def train_mnist(config, reporter):
    # https://github.com/tensorflow/tensorflow/issues/32159
    batch_size = 128
    num_classes = 10
    epochs = 12

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(config["hidden"], activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.SGD(
            lr=config["lr"], momentum=config["momentum"]),
        metrics=["accuracy"])

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(x_test, y_test),
        callbacks=[TuneReporterCallback(reporter)])


if __name__ == "__main__":
    import ray
    from ray import tune
    from ray.tune.schedulers import AsyncHyperBandScheduler

    # client = MlflowClient()
    # experiment_id = client.create_experiment("test7")

    mnist.load_data()  # we do this on the driver because it's not threadsafe

    ray.init()
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="mean_accuracy",
        mode="max",
        max_t=400,
        grace_period=20)

    # tf.keras.mlflow.MLFLowLogger()
    tune.run(
        train_mnist,
        name="mlflow",
        scheduler=sched,
        stop={
            "mean_accuracy": 0.99,
            "training_iteration": 5 if args.smoke_test else 300
        },
        num_samples=10,
        resources_per_trial={
            "cpu": 2,
            "gpu": 0
        },
        loggers=DEFAULT_LOGGERS + (MLFLowLogger, ),
        config={
            # "mlflow_experiment_id": experiment_id,
            "threads": 2,
            "lr": tune.sample_from(lambda spec: np.random.uniform(0.001, 0.1)),
            "momentum": tune.sample_from(
                lambda spec: np.random.uniform(0.1, 0.9)),
            "hidden": tune.sample_from(
                lambda spec: np.random.randint(32, 512)),
        })