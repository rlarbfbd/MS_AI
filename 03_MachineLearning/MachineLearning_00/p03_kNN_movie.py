import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import mlflow

mlflow.start_run()

feature = np.array([[80, 20], [95, 5], [10, 90], [90, 10], [5, 95]])
label = np.array(["액션", "액션", "조폭", "액션", "조폭"])

knc = KNeighborsClassifier(3)                       # k = 3: 가장 가까운 데이터 3개를 찾겠다
knc.fit(feature, label)                             # 학습시키기

fight = 30
yok = 50
z = np.array([[fight, yok]])                        # 예측할 데이터를 feature랑 같은 형태로
result = knc.predict(z)                             # 예측
print(result)

mlflow.sklearn.save_model(sk_model=knc, path="./Users/stu_ict01_01/May28_3_MachineLearning/kNN_movie")

mlflow.end_run()
