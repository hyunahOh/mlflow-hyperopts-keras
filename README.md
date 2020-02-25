# Deep face recognition with Keras, Dlib and OpenCV

This repository hosts the [companion notebook](http://nbviewer.jupyter.org/github/krasserm/face-recognition/blob/master/face-recognition.ipynb?flush_cache=true) to the article [Deep face recognition with Keras, Dlib and OpenCV](https://krasserm.github.io/2018/02/07/deep-face-recognition/).

## Face embedding
`embedding.py` : Openface pretrained model을 가져와서, 128차원 벡터값으로 변환 후, 저장함

## Classification (training)
`classification.py` : 각 사람마다 128벡터화 되어있는 정보를 분류 학습 -> `mlflow`와 `hyperopt`가 적용되어있는 코드