# 数据读取
import pandas as pd
import numpy as np
import random

# 预处理
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
# 导入计算指标
from sklearn import metrics

data = pd.read_csv("AAA.csv")
data_pred = pd.read_csv("BBB.csv")

# LinearRegression
randint = random.randint(1, 100)

data_train, data_test = train_test_split(data, random_state=randint, test_size=0.2)
X_train = data_train.drop(labels=['output1', 'output2'], axis=1)
X_test = data_test.drop(labels=['output1', 'output2'], axis=1)
X_pred = data_pred.drop(labels=['output1', 'output2'], axis=1)
print('训练集\n', X_train, '\n------------------------\n测试集', X_test, '\n------------------------\n预测集', X_pred)
print('type(X_train)', type(X_train), '\ntype(X_test)', type(X_test), '\ntype(X_pred)', type(X_pred))
standardscaler = preprocessing.StandardScaler()
standardscaler.fit(X_train)

X_train = standardscaler.transform(X_train)
X_test = standardscaler.transform(X_test)
X_pred = standardscaler.transform(X_pred)

print('type(X_train)', type(X_train), '\ntype(X_test)', type(X_test), '\ntype(X_pred)', type(X_pred))

y_train = (data_train.loc[:, ['output1']]).values.ravel()
y_test = (data_test.loc[:, ['output1']]).values.ravel()
print('y_train', y_train, '\ntype(y_train)', type(y_train), '\ny_test', y_test,'\ntype(y_test)', type(y_test))

clf = LinearRegression()
model = clf.fit(X_train, y_train)

print(X_test)
y_pred = model.predict(X_test)
print(y_pred)

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

# 第二个输出-----------------------------------------------------------------
# 第二个输出-----------------------------------------------------------------
# 第二个输出-----------------------------------------------------------------
# 第二个输出-----------------------------------------------------------------
# 第二个输出-----------------------------------------------------------------
# 第二个输出-----------------------------------------------------------------
# 第二个输出-----------------------------------------------------------------

y_train = (data_train.loc[:, ['output2']]).values.ravel()
y_test = (data_test.loc[:, ['output2']]).values.ravel()
print('y_train', y_train, '\ntype(y_train)', type(y_train), '\ny_test', y_test, '\ntype(y_test)', type(y_test))

clf = LinearRegression()
model = clf.fit(X_train, y_train)

print(X_test)
y_pred = model.predict(X_test)
print(y_pred)

yy_pred = model.predict(X_pred)
print('yy', yy_pred)

print("结果如下：")
print("训练集分数：", model.score(X_train, y_train))
print("验证集分数：", model.score(X_test, y_test))

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)
print(mae)
print(mse)
print(r2)