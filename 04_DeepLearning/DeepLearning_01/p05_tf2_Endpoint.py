import mlflow.tensorflow
import tensorflow as tf
import numpy as np
import mlflow

mlflow.start_run()

xData = np.array([[80, 20], [95, 5], [10, 90], [90, 10], [5, 95]])
yData = np.array([[1, 0], [1, 0], [0, 1], [1, 0], [0, 1]])
label = ["액션", "조폭"]

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(2, 1)), 
        tf.keras.layers.Dense(100, activation="relu"),    
        tf.keras.layers.Dense(200, activation="relu"),  
        tf.keras.layers.Dense(50, activation="relu"),         
        tf.keras.layers.Dense(2, activation="softmax"),       
    ]
)

o = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=o, loss="categorical_crossentropy")

model.fit(xData, yData, epochs=1000)

f = 90
y = 10
predictData = np.array([[f, y]])

result = model.predict(predictData)
print(result)
print(label[result[0].argmax()])

mlflow.tensorflow.save_model(model=model, path=".Users/stu_ict01_01/Jun02_1_DeepLearning/rosalie")

mlflow.end_run()


