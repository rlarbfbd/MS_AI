# %%
import tensorflow as tf
import numpy as np
import cv2
import os

# a = cv2.imread("./Users/stu_ict01_01/Jun02_1_DeepLearning/bunsikMenu/i01.png", cv2.IMREAD_GRAYSCALE)
# print(a)          # 테스트

# %%
def getImg(folder, w, h):  
    datas = []                     
    for f in os.listdir(folder):                # 사진 가로,세로 = 행열
        data = cv2.imread(folder + "/" + f, cv2.IMREAD_GRAYSCALE)
        data = cv2.resize(data, (w, h))
        datas.append(data)
    return np.array(datas)

# %%
# 인공신경망: 행렬계산 + 회귀 -> 데이터가 숫자여야함
# yData = one-hot encoding
# xData = 숫자/글자/이미지/음성/... -> 어차피 컴에서는 다 전기신호(2진수) -> 숫자
# 데이터 인코딩/디코딩만 한다면
xData = getImg("./Users/stu_ict01_01/Jun02_1_DeepLearning/bunsikMenu", 100, 50)
# yData = [[1,0,0,0,0], [1,0,0,0,0], ...]  => 실제 라벨링해서 넣는 데이터
yData = [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4]         
yData = tf.one_hot(yData, 5)                        # tensorflow의 기능을 활용해서 편하게 라벨링
label = ["떡볶이", "오뎅", "김밥", "튀김", "순대"]
'''
print(len(xData))
print(len(xData[0]))        # 세로
print(len(xData[0][0]))     # 가로
'''

# %%
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(50, 100)),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),         
        tf.keras.layers.Dense(5, activation="softmax")      
    ]
)

o = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=o, loss="categorical_crossentropy")

# %%
model.fit(xData, yData, epochs=1000)

# %%
predictData = cv2.imread("./Users/stu_ict01_01/Jun02_1_DeepLearning/test2.png", cv2.IMREAD_GRAYSCALE)
predictData = cv2.resize(predictData, (100, 50))
predictData = np.array([predictData])

result = model.predict(predictData)
print(label[result[0].argmax()])

# %%



