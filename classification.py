'''This code is for Training face recognition
 author : Hyunah Oh
 data : 2020.01.22
 flow : Load Embedded code -> Training Classification
'''

import numpy as np
np.random.seed(2020)
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from data import load_metadata

from hyperas.distributions import uniform, choice
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp

from keras.models import Sequential
from keras.optimizers import SGD,RMSprop
from keras.layers import Dense, Dropout, Activation

import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient

# mlflow 자동로그
mlflow.keras.autolog()

# Hyperparameter search space
search_space = {
    "optimizer": hp.choice('optmz',["sgd", "rms"]),
    "epochs": hp.choice('epochs', [30, 40, 50, 60, 70]),
    "lr": hp.uniform('lr',0,1)
}

def data():
    '''
        데이터를 준비하여 train/val/test로 나눠주는 함수
    '''
    metadata = load_metadata('images')
    targets = np.array([m.name for m in metadata])
    embedded = joblib.load('models/embedded_images.pkl')

    encoder = LabelEncoder()
    encoder.fit(targets)

    # Numerical encoding of identities
    y = encoder.transform(targets)

    train_idx = np.arange(metadata.shape[0]) % 3 == 2
    val_idx = np.arange(metadata.shape[0]) % 3 == 1
    test_idx = np.arange(metadata.shape[0]) % 3 == 0

    # 40 train examples of 4 identities (10 examples each)
    X_train = embedded[train_idx]    
    X_val = embedded[val_idx]
    X_test = embedded[test_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_model_hypopt(params):
    '''
        모델을 만들고 학습하는 부분
    '''
    X_train, y_train, X_val, y_val, X_test, y_test = data()
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(128,)))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    with mlflow.start_run(experiment_id=3):
        # 모델 컴파일 하기
        lr = params["lr"]
        epochs = params["epochs"]
        if params["optimizer"] == 'rms':
            optimizer = RMSprop(lr=lr)
        else:
            optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

        # mlflow 수동 로그
        mlflow.log_param('lr', lr)
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('optimizer', optimizer)

        model.compile(optimizer=optimizer, #rmsprop
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

        # fit() 메서드로 모델 훈련 시키기
        history = model.fit(X_train, y_train, epochs=30, batch_size=128, validation_data=(X_val, y_val))

        # 테스트 데이터로 정확도 측정하기
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print('test_acc: ', test_acc)
        mlflow.log_metric('acc', test_acc)

        val_loss = np.amax(history.history['val_loss'])
        print('Best validation error of epoch:', val_loss)
        return {'loss': val_loss, 'status': STATUS_OK, 'model': model} # if accuracy use '-' sign
        # return history, lstm_model 

if __name__=="__main__":
    client = MlflowClient()
    experiment_id = client.create_experiment("tmp")

    trials = Trials()
    best = fmin(create_model_hypopt,
    space=search_space,
    algo=tpe.suggest, # type random.suggest to select param values randomly
    max_evals=200, # max number of evaluations you want to do on objective function
    trials=trials)

    joblib.dump(bets['model'], 'models/classification_model.pkl')