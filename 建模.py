import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin


class DenseTransformer(BaseEstimator, TransformerMixin):
    """将稀疏矩阵转换为密集矩阵的转换器"""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, 'toarray'):
            return X.toarray()
        return X


def load_and_preprocess_data(filepath):
    """加载数据并进行预处理"""
    data = pd.read_csv(filepath)

    # 选择关键列
    critical_cols = ['City Name', 'Package', 'Variety', 'Date', 'Low Price', 'High Price']
    data = data[critical_cols].copy()

    # 创建目标变量（使用Low Price）
    data['target'] = data['Low Price']

    # 处理日期特征
    data['Date'] = pd.to_datetime(data['Date'])
    data['year'] = data['Date'].dt.year
    data['month'] = data['Date'].dt.month
    data['day'] = data['Date'].dt.day

    return data.drop(columns=['Date', 'Low Price', 'High Price'])


def build_model_pipeline():
    """构建处理NaN值的模型管道"""
    # 数值特征处理
    numeric_features = ['year', 'month', 'day']
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )

    # 分类特征处理
    categorical_features = ['City Name', 'Package', 'Variety']
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore', sparse_output=True),
        DenseTransformer()  # 添加稀疏矩阵转换
    )

    # 组合预处理步骤
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 创建两种模型管道
    models = {
        'LinearRegression': make_pipeline(
            preprocessor,
            SimpleImputer(strategy='mean'),  # 额外确保没有NaN
            LinearRegression()
        ),
        'HistGradientBoosting': make_pipeline(
            preprocessor,
            HistGradientBoostingRegressor(random_state=42)
        )
    }

    return models


def evaluate_models(models, X_train, X_test, y_train, y_test):
    """评估并比较不同模型"""
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            'Model': name,
            'MSE': mean_squared_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        })

    return pd.DataFrame(results)


# 主程序
if __name__ == "__main__":
    # 加载数据
    data = load_and_preprocess_data('US-pumpkins.csv')

    # 准备特征和目标
    X = data.drop(columns=['target'])
    y = data['target']

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 构建和评估模型
    models = build_model_pipeline()
    results = evaluate_models(models, X_train, X_test, y_train, y_test)

    # 显示结果
    print("模型评估结果:")
    print(results.sort_values('R2', ascending=False))
