# Data Load & Preprocessing
"""
    Import
    - Sequential      : 레이어를 위에서 아래로 순서대로 쌓는 가장 단순한 모델 컨테이너, input -> hidden -> output 같은 직렬 구조에 쓰임
    - Dense           : Fully connected layer의 한 종류, 각 뉴런이 이전 레이어의 모든 뉴런과 연결됨
    - fahion_mnist    : keras에 있는 fahion_mnist 데이터셋 로드
    - to_categorical  : output label(0-9)을 원-핫 벡터로 변환해줌
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

"""
    Constants
    - Reshaped = 784  : 데이터 사진 한장은 28*28=784 픽셀, Dense layer가 2D 이미지를 그대로 받지 못하여 일렬로 펼친 벡터로 넣어줘야 함
    - NB_CLASSES = 10 : 데이터의 클래스가 총 10개라서 출력층 뉴런도 10개
"""
RESHAPED = 784
NB_CLASSES = 10

"""
    Load Fashion-MNIST
    - x는 이미지, y는 정답 라벨
"""
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

"""
    Preprocessing 
    - train과 test가 같은 방식으로 전처리되어야 공정
    - reshape(-1,784) : 원래 x_train은 (60000,28,28)이지만 (60000,784)로 변환, -1은 개수는 알아서 맞춰라는 지시
    - astype("float32): 픽셀 값은 보통 정수형인데 학습은 실수형이 안정적이라 float로 변환
    - /255            : 픽셀값이 0-255에서 0-1로 정규화 -> 학습이 훨씬 안정적이고 빠르게 수렴함
"""
x_train = x_train.reshape(-1, RESHAPED).astype("float32") / 255
x_test = x_test.reshape(-1, RESHAPED).astype("float32") / 255

"""
    - 라벨을 원-핫으로 변환
    - 이 작업으로 categorical_crossentropy loss function을 사용가능
"""
y_train = to_categorical(y_train, NB_CLASSES)
y_test = to_categorical(y_test, NB_CLASSES)

print(x_train.shape, y_train.shape)


# Baseline(20s)
"""
    - model.add()를 이용해 한 층씩 쌓아올리기
    - model.add(Dense): Dense layer 한 층 추가, 뉴런 64개, 
                        activation function은 ReLU(음수는 0으로, 양수는 그대로),
                        input_shape(RESHAPED,)는 입력이 784짜리 벡터라는 뜻
    - softmax         : 출력층, 뉴런 10개,
                        10개 값이 확률분포처럼 합이 1이 되게 변환 -> 각 클래스일 확률
"""

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(RESHAPED,)))
model.add(Dense(NB_CLASSES, activation='softmax'))

"""
    - compile(학습 규칙 세팅)
    - optimizer= : 가중치를 어떻게 업데이트할지(학습 방법), 가장 기본적 세팅
    - loss=      : 정답과 결과의 차이를 계산하는 손실함수,
                   다중분류에 표준
    - metrics=   : 학습중에 accuracy도 같이 출력해달라는 뜻,
                   loss만으로는 직관이 떨어지기 때문에 accuracy도 참고하여 해석
"""
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

"""
    - epochs=           : 전체 데이터 10번 반복 학습
    - batch_size=       : 한번에 128장씩 묶어서 계산(클수록 속도↑, 메모리↑, 업데이트↓)
    - validation_split  : 학습데이터를 10% 떼서 검증용으로 사용(과적합 판단에 좋음)
    - history           : 매 epoch마다 loss/acc 기록이 들어있음
"""
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)



# Deep Network(32s) - baseline과 같지만 두개의 은닉층 추가
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(RESHAPED,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(NB_CLASSES, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)


# Wide Network(1m 29s) - 뉴런 수를 512개로 늘려 한 번에 더 많은 특징 조합을 학습(연산량 폭증 -> 러닝타임 폭증)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(RESHAPED,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(NB_CLASSES, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)
