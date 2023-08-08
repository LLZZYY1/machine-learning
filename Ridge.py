# 数据读取
import pandas as pd
# import numpy as np
import random

# 预处理
from sklearn import preprocessing
from sklearn.model_selection import train_test_split  # KFold
# Ridge
from sklearn.linear_model import Ridge

# 导入计算指标
from sklearn import metrics

data = pd.read_csv("AAA.csv")
data_predict = pd.read_csv("BBB.csv")

randint = random.randint(1, 100)
data_train, data_test = train_test_split(data, random_state=randint, test_size=0.2)
print(data_train, data_test)
X_train = data_train.drop(labels=['output1', 'output2'], axis=1)
X_test = data_test.drop(labels=['output1', 'output2'], axis=1)
X_predict = data_predict.drop(labels=['output1', 'output2'], axis=1)

standardscaler = preprocessing.StandardScaler()
standardscaler.fit(X_train)
X_train = standardscaler.transform(X_train)
X_test = standardscaler.transform(X_test)
X_predict = standardscaler.transform(X_predict)

y_train = (data_train.loc[:, ['output1']]).values.ravel()
y_test = (data_test.loc[:, ['output1']]).values.ravel()

clf = Ridge()
model = clf.fit(X_train, y_train)
y_predict = model.predict(X_test)

yy_predict = model.predict(X_predict)
print(yy_predict)


print("结果如下：")
print("训练集分数：", model.score(X_train, y_train))
print("验证集分数：", model.score(X_test, y_predict))
mae = metrics.mean_absolute_error(y_test, y_predict)
mse = metrics.mean_squared_error(y_test, y_predict)
r2 = metrics.r2_score(y_test, y_predict)
print(mae)
print(mse)
print(r2)
