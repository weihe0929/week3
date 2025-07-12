import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib import rcParams
#设置字体为支持中文
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
# 加载数据集
df = pd.read_csv('US-pumpkins.csv')
# 数据清洗与预处理
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Year'] = df['Date'].dt.year
# 计算价格中位数
df['价格中位数'] = (df['Low Price'] + df['High Price']) / 2
# 分类变量处理
分类列 = ['City Name', 'Package', 'Variety', 'Sub Variety', 'Grade', 'Origin',
         'Item Size', 'Color', 'Environment', 'Unit of Sale', 'Quality',
         'Condition', 'Appearance', 'Storage', 'Crop', 'Repack', 'Trans Mode']
# 对分类变量进行独热编码
df_编码 = pd.get_dummies(df, columns=分类列)

# 选择数值型列进行相关性分析
数值列 = df_编码.select_dtypes(include=[np.number]).columns.tolist()
df_数值 = df_编码[数值列]
# 计算相关性矩阵
相关性矩阵 = df_数值.corr()
# 绘制热力图
plt.figure(figsize=(20, 15))
sns.heatmap(相关性矩阵, cmap='coolwarm', center=0, annot=False,
            xticklabels=False, yticklabels=False)
plt.title('南瓜价格特征相关性热力图', fontsize=16)
plt.tight_layout()
plt.show()
# 附加分析 - 按品种和月份的价格趋势
plt.figure(figsize=(12, 6))
sns.boxplot(x='Month', y='价格中位数', hue='Variety', data=df)
plt.title('不同月份和品种的南瓜价格分布', fontsize=14)
plt.ylabel('价格中位数(美元)')
plt.xlabel('月份')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# 按产地的价格分析
plt.figure(figsize=(12, 6))
top_origins = df['Origin'].value_counts().nlargest(10).index
df_top_origins = df[df['Origin'].isin(top_origins)]
sns.boxplot(x='Origin', y='价格中位数', data=df_top_origins)
plt.title('前10产地南瓜价格分布', fontsize=14)
plt.ylabel('价格中位数(美元)')
plt.xlabel('产地')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# 包装类型分析
plt.figure(figsize=(12, 6))
sns.boxplot(x='Package', y='价格中位数', data=df)
plt.title('不同包装类型的南瓜价格分布', fontsize=14)
plt.ylabel('价格中位数(美元)')
plt.xlabel('包装类型')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()