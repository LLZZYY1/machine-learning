# 数据读取
import pandas as pd
import numpy as np
import random

# 预处理
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
# 导入计算指标
from sklearn import metrics

data = pd.read_csv("AAA.csv")
data_pred = pd.read_csv("BBB.csv")

# BaggingRegressor
from sklearn.ensemble import BaggingRegressor

randint = random.randint(1, 100)
data_train, data_test = train_test_split(data, random_state=randint, test_size=0.2)
X_train = data_train.drop(labels=['output1', 'output2'], axis=1)
X_test = data_test.drop(labels=['output1', 'output2'], axis=1)
X_pred = data_pred.drop(labels=['output1', 'output2'], axis=1)

standardscaler = preprocessing.StandardScaler()
standardscaler.fit(X_train)

X_train = standardscaler.transform(X_train)
X_test = standardscaler.transform(X_test)
X_pred = standardscaler.transform(X_pred)

y_train = (data_train.loc[:, ['output1']]).values.ravel()
y_test = (data_test.loc[:, ['output1']]).values.ravel()

clf = BaggingRegressor()
model = clf.fit(X_train, y_train)
y_pred = model.predict(X_test)

yy_pred = model.predict(X_pred)
print(yy_pred)

print("结果如下：")
print("训练集分数：", model.score(X_train, y_train))
print("验证集分数：", model.score(X_test, y_test))

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)
print(mae)
print(mse)
print(r2)
