'''
import matplotlib.pyplot as plt
import numpy as np

#画个sin函数
x = np.linspace(0, 2 * np.pi, 50)
y = np.sin(x)
# plt.plot(x, y)
# plt.show()

#绘制两个图像
# plt.plot(x, y)
# plt.plot(x, y * 2)
# plt.show()
#
# #构建子图
# ax1 = plt.subplot(2, 1, 1) # （行，列，活跃区）
# plt.plot(x, np.sin(x), 'r')
#
# ax2 = plt.subplot(2, 3, 4)
# plt.plot(x, 2 * np.sin(x), 'g')
#
# ax3 = plt.subplot(2, 3, 5, sharey=ax2)
# plt.plot(x, np.cos(x), 'b')
#
# ax4 = plt.subplot(2, 3, 6, sharey=ax2)
# plt.plot(x, 2 * np.cos(x), 'y')
#
# plt.show()

#散点图
# k = 500
# x = np.random.rand(k)
# y = np.random.rand(k)
# size = np.random.rand(k) * 50 # 生成每个点的大小
# colour = np.arctan2(x, y) # 生成每个点的颜色大小
# plt.scatter(x, y, s=size, c=colour)
# plt.colorbar() # 添加颜色栏
#
# plt.show()

#柱状图
k = 10
x = np.arange(k)
y = np.random.rand(k)
plt.bar(x, y) # 画出 x 和 y 的柱状图
# 增加数值
for x, y in zip(x, y):
    plt.text(x, y , '%.2f' % y, ha='center', va='bottom')
plt.show()

# import seaborn as sns
# data = sns.load_dataset("iris")
'''


# Exercise: size analysis of iris flower
'''
Requirement：
	1.Size relationship between sepal and petal (scatter diagram)
 	2.The size relationship between sepals and petals of iris of different species
 	3.Distribution of sepals and petal sizes of different Iris species (box diagram)
'''


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import seaborn as sb
import matplotlib.cm as cm


iris = load_iris()
X, y = iris.data, iris.target

# import data of iris flower
iris_data = pd.DataFrame(np.hstack((X, y.reshape(-1, 1))),index = range(X.shape[0]),columns=['Sepal_Len(cm)','sepal_wid(cm)','Petal_Len(cm)','petal_wid(cm)','Species(class)'] )
print(iris_data.describe())  # check the data

sb.pairplot(iris_data.dropna(),hue = 'Species(class)')
plt.figure(figsize=(10,10))
for column_index,column in enumerate(iris_data.columns):
    if column == 'Species(class)':
        continue
    plt.subplot(2,2,column_index+1)
    sb.boxplot(x= 'Species(class)', y=column, data = iris_data)

plt.show()









































