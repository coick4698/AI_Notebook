"""
  - Sequential: FCN 모델 만들 때 사용
  - Dense: Fully Connected layer
  - fashion_mnist: 사용할 dataset
  - to_cotegorical: one-hot encoding
"""
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam, RMSprop

"""
  Constant Justification
  - Reshaped: 28*28 image size
  - NB_Classes: 10개의 class
"""
RESHAPED = 784
NB_CLASSES = 10

"""
  Data Load
  - x_train: 훈련 이미지
  - y_train: 훈련 라벨
  - x_test: 테스트 이미지
  - y_test: 테스트 라벨
"""
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

"""
  Image Preprocessing(***)
  - reshape: FCN은 1차원 벡터만 입력받기 때문에 28*28을 784로 치환; (60000, 28, 28) -> (60000, 784)
  - astype: 정수 -> 실수형 변환
  - /255.0: 픽셀 값 0-255를 0-1 사이로 정규화; 학습 안정화 목적
"""
x_train = x_train.reshape(-1, RESHAPED).astype('float32') / 255.0
x_test = x_test.reshape(-1, RESHAPED).astype('float32') / 255.0

"""
  One-hot encoding
  - why? softmax + categorical_crossentropy를 쓰기 때문에
"""
y_train = to_categorical(y_train, NB_CLASSES)
y_test = to_categorical(y_test, NB_CLASSES)



# Base Model

"""
  - 입력층 + 은닉층1 Dense :  hidden_units=128: 뉴런 128개(default),
                          input_shape(RESHAPED,): 입력 벡터 길이가 784라는 뜻,
                          relu: 은닉층 activation function(비선형성 추가)
                          -> 784차원 입력을 128차원 특징으로 압축 및 변환
  - 은닉층2 Dense        : 똑같은 128개 뉴런 층을 더 쌓아서 더 복잡한 패턴을 학습할 용량을 늘리는 구조
  - 출력층 Dense         : NB_CLASSES=10: 10개 class 확률 출력,
                         softmax: 10개 class에 대한 확률 분포로 변환(총합 1)
  - compile            : optimizer=: SGD/ADAM 등을 커스텀할 수 있게 함수 인자로 설정,
                         loss=: 다중 분류에서 표준 loss function,
                         metrics=[]: 학습 중 정확도도 같이 출력/기록
                
"""
def build_model(optimizer, hidden_units=128):
  model = Sequential()
  model.add(Dense(hidden_units, input_shape=(RESHAPED,), activation='relu'))
  model.add(Dense(hidden_units, activation='relu'))
  model.add(Dense(NB_CLASSES, activation='softmax'))

  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model



# Experiment 1: Tuning Learning Rate(SGD)
"""
  - learning_rates: 4개의 후보
  - results_lr={}: 나중에 lr -> test accuracy를 저장할 딕셔너리

  - for loop: 리스트에 있는 lr 후보를 하나씩 반복 실험
  - print: 어떤 lr로 실험 중인지 로그 출력
  - optimizer=: 같은 SGD라도 학습률만 바꿔가며 성능 차이를 보이게 하는 설정 ***
                lr 크면 업데이트 큼(빠르지만 불안정),
                lr 작으면 업데이트 작음(안정적이지만 느림)
  - model=: base model과 동일한 구조의 모델 생성
  - history=: 128개 샘플로 gradient 계산 후 업데이트,
              전체 데이터를 10번 반복 학습,
              train data 중 20%를 검증용으로 나누어 val_accuracy 추적,
              verbose를 0으로 진행 로그 출력 안함
  - test_loss: 학습에 안 쓴 x_test, y_test로 성능 확인
  - test_acc: lr 비교의 최종 기준 점수
  - print: 4자리 소수로 출력 후 딕셔너리 저장
  
"""
learning_rates = [0.1, 0.01, 0.001, 0.0001]
results_lr = {}

for lr in learning_rates:
  print(f"Training with learning rate: {lr}")
  optimizer = SGD(learning_rate=lr)
  model = build_model(optimizer)
  history = model.fit(
                      x_train, y_train,
                      batch_size=128,
                      epochs=10,
                      validation_split=0.2,
                      verbose=0)
  test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
  results_lr[lr] = test_acc
  print(f"Test accuracy for lr={lr}: {test_acc:.4f}")


 # Visualising
"""
  - history: model.fit()의 return 값
  - x축: epoch(반복 횟수)
  - y축: accuracy(정확도 0-1)
  - 파란선: train accuracy(train data에서 얼마나 잘 맞추는지)
  - 주황선: validation accuracy(validation data에서 얼마나 잘 맞추는지
  -> Overfitting 판단기준: train accuracy 계속 상승, 
                         validation accuracy가 특정 시점부터 하락,
                         둘 사이 간격 커짐                       
  -> Underfitting 판단기준: train accuracy와 validation accuracy 둘 다 낮음
"""

import matplotlib.pyplot as plt

def plot_history(history):
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

plot_history(history)



# Experiment 2: ADD Momentum in SGD

"""
  - SGD: 매 step마다 방향을 새로 결정
  - SGD + Momentum: 이전 방향을 기억해서 진동 줄이고 더 빠르게 내려감(관성)
"""

momentum_values = [0.0, 0.5, 0.9]
results_momentum = {}

for mom in momentum_values:
    print(f"Training with momentum: {mom}")

    optimizer = SGD(learning_rate=0.01, momentum=mom)
    model = build_model(optimizer)

    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=10,
        validation_split=0.2,
        verbose=0
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    results_momentum[mom] = test_acc
    print(f"Test accuracy for momentum={mom}: {test_acc:.4f}")



# Experiment 3: Comparing Optimizers

"""
  - SGD: default,
         learning rate에 가장 민감,
         튜닝 안하면 성능 저하
         -> 잘 쓰면 강력하지만 다루기 어려움
  - ADAM: Momentum + Adaptive Learning Rate,
          파라미터마다 다른 learning rate 적용,
          튜닝 없이도 좋은 성능, default
          -> 방향 기억함, 각 weight마다 자동 조정
  - RMSprop: gradient 크기에 따라 학습률 자동 조절,
             RNN이나 deeper network에서 강점
             -> SGD보다 안정적, ADAM과 비슷하지만 구조 다름
"""

optimizers = {
    'SGD': SGD(learning_rate=0.01),
    'Adam': Adam(learning_rate=0.001),
    'RMSprop': RMSprop(learning_rate=0.001, momentum=0.9)
}

results_opt = {}

for name, opt in optimizers.items():
    print(f"Training with optimizer: {name}")

    model = build_model(opt)

    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=10,
        validation_split=0.2,
        verbose=0
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    results_opt[name] = test_acc
    print(f"Test accuracy for {name}: {test_acc:.4f}")




