import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
from matplotlib import rcParams
#设置字体为支持中文
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 配置全局参数
class Config:
    DATE_COL = 'Date'
    TARGET_COL = 'PriceMedian'
    RANDOM_STATE = 42
    TS_CV_SPLITS = 5
    PLOT_STYLE = 'seaborn-v0_8'
    FIG_SIZE = (12, 6)
    USE_PARALLEL = True  # 全局并行控制
    N_JOBS = -1  # 并行工作数


# 忽略警告
warnings.filterwarnings('ignore')
plt.style.use(Config.PLOT_STYLE)


class DataLoader:
    """智能数据加载与预处理"""

    @staticmethod
    def load_data(filepath):
        """优化后的数据加载管道"""
        df = (
            pd.read_csv(filepath, parse_dates=[Config.DATE_COL])
            .pipe(DataLoader._basic_cleaning)
            .pipe(DataLoader._add_temporal_features)
            .pipe(DataLoader._handle_outliers)
            .pipe(DataLoader._encode_categories)
        )
        return df

    @staticmethod
    def _basic_cleaning(df):
        """基础数据清洗"""
        return df.assign(
            PriceMedian=lambda x: (x['High Price'] + x['Low Price']) / 2,
            Package=lambda x: x['Package'].str[:30]  # 截断长文本
        ).dropna(subset=['PriceMedian'])

    @staticmethod
    def _add_temporal_features(df):
        """添加时序特征"""
        return df.assign(
            Year=lambda x: x[Config.DATE_COL].dt.year,
            Month=lambda x: x[Config.DATE_COL].dt.month,
            DayOfYear=lambda x: x[Config.DATE_COL].dt.dayofyear,
            WeekOfYear=lambda x: x[Config.DATE_COL].dt.isocalendar().week,
            Month_sin=lambda x: np.sin(2 * np.pi * x[Config.DATE_COL].dt.month / 12),
            Month_cos=lambda x: np.cos(2 * np.pi * x[Config.DATE_COL].dt.month / 12),
            DayOfWeek=lambda x: x[Config.DATE_COL].dt.dayofweek,
            IsWeekend=lambda x: x[Config.DATE_COL].dt.dayofweek.isin([5, 6]).astype(int)
        )

    @staticmethod
    def _handle_outliers(df):
        """处理异常值"""
        q_low = df['PriceMedian'].quantile(0.01)
        q_high = df['PriceMedian'].quantile(0.99)
        return df[df['PriceMedian'].between(q_low, q_high)]

    @staticmethod
    def _encode_categories(df):
        """分类变量预处理"""
        df['Variety'] = df['Variety'].where(
            df['Variety'].map(df['Variety'].value_counts()) > 50,
            'Other'
        )
        return df


class FeatureAnalyzer:
    """特征分析与可视化"""

    @staticmethod
    def plot_feature_distributions(df):
        """绘制特征分布"""
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        plt.figure(figsize=Config.FIG_SIZE)
        sns.pairplot(df[num_cols[:5]], diag_kind='kde')
        plt.suptitle('数值特征分布矩阵', y=1.02)
        plt.show()

        plt.figure(figsize=Config.FIG_SIZE)
        sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('特征相关性矩阵')
        plt.show()

    @staticmethod
    def analyze_temporal_trends(df):
        """分析时序趋势"""
        fig, ax = plt.subplots(2, 1, figsize=(Config.FIG_SIZE[0], Config.FIG_SIZE[1] * 1.5))

        # 月度趋势
        monthly = df.groupby('Month')['PriceMedian'].agg(['mean', 'std'])
        monthly.plot(kind='bar', y='mean', yerr='std', ax=ax[0], capsize=4)
        ax[0].set_title('月度价格趋势')

        # 品种趋势
        sns.boxplot(x='Month', y='PriceMedian', hue='Variety',
                    data=df[df['Variety'].isin(df['Variety'].value_counts().index[:3])],
                    ax=ax[1])
        ax[1].set_title('主要品种月度价格分布')
        plt.tight_layout()
        plt.show()


class FeatureProcessor:
    """特征工程处理器"""

    @staticmethod
    def get_feature_pipeline():
        """创建特征处理管道"""
        # 数值特征处理
        num_pipeline = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler()
        )

        # 分类特征处理
        cat_pipeline = make_pipeline(
            SimpleImputer(strategy='most_frequent'),
            OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False),
            StandardScaler()
        )

        # 自动特征选择
        preprocessor = ColumnTransformer([
            ('num', num_pipeline, make_column_selector(dtype_include=np.number)),
            ('cat', cat_pipeline, ['Variety', 'Package', 'City Name'])
        ])

        return preprocessor


class ModelBuilder:
    """模型构建与评估"""

    @staticmethod
    def _get_parallel_params():
        """获取并行计算参数"""
        return {'n_jobs': Config.N_JOBS} if Config.USE_PARALLEL else {}

    @staticmethod
    def build_models():
        """构建模型集合"""
        base_params = {
            'random_state': Config.RANDOM_STATE
        }

        parallel_params = ModelBuilder._get_parallel_params()

        return {
            'HGB': HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=200,
                **base_params
            ),
            'XGB': XGBRegressor(
                n_estimators=150,
                subsample=0.8,
                tree_method='hist',
                **base_params,
                **parallel_params
            ),
            'LGBM': LGBMRegressor(
                n_estimators=120,
                subsample=0.7,
                **base_params,
                **parallel_params
            ),
            'Ensemble': VotingRegressor([
                ('hgb', HistGradientBoostingRegressor(**base_params)),
                ('xgb', XGBRegressor(**base_params, **parallel_params))
            ])
        }

    @staticmethod
    def evaluate_model(model, X, y):
        """评估模型性能"""
        tscv = TimeSeriesSplit(n_splits=Config.TS_CV_SPLITS)
        pipeline = make_pipeline(FeatureProcessor.get_feature_pipeline(), model)

        metrics = {
            'MAE': cross_val_score(pipeline, X, y, cv=tscv,
                                   scoring='neg_mean_absolute_error'),
            'R2': cross_val_score(pipeline, X, y, cv=tscv, scoring='r2')
        }

        return {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in metrics.items()}

    @staticmethod
    def plot_feature_importance(model, feature_names):
        """可视化特征重要性"""
        if hasattr(model, 'feature_importances_'):
            importance = pd.Series(
                model.feature_importances_,
                index=feature_names
            ).sort_values(ascending=False)

            plt.figure(figsize=Config.FIG_SIZE)
            importance.head(15).plot(kind='barh')
            plt.title('Top 15 重要特征')
            plt.show()


def main():
    # 1. 数据加载
    print("⏳ 加载数据...")
    df = DataLoader.load_data('US-pumpkins.csv')
    print(f"✅ 数据加载完成，共 {len(df)} 条记录")

    # 2. 特征分析
    print("\n🔍 分析特征分布...")
    FeatureAnalyzer.plot_feature_distributions(df)
    FeatureAnalyzer.analyze_temporal_trends(df)

    # 3. 准备建模数据
    X = df.drop(columns=[Config.TARGET_COL, Config.DATE_COL])
    y = df[Config.TARGET_COL]

    # 4. 模型训练与评估
    print("\n🤖 训练模型中...")
    models = ModelBuilder.build_models()
    results = []

    for name, model in models.items():
        scores = ModelBuilder.evaluate_model(model, X, y)
        results.append({
            'Model': name,
            'MAE': f"{abs(scores['MAE']['mean']):.2f} ±{scores['MAE']['std']:.2f}",
            'R2': f"{scores['R2']['mean']:.3f} ±{scores['R2']['std']:.3f}"
        })

    # 5. 结果展示
    results_df = pd.DataFrame(results)
    print("\n📊 模型性能对比:")
    print(results_df.sort_values('R2', ascending=False))

    # 6. 最佳模型分析
    best_model = make_pipeline(
        FeatureProcessor.get_feature_pipeline(),
        models['Ensemble']
    )
    best_model.fit(X, y)

    try:
        feature_names = best_model[0].get_feature_names_out()
        ModelBuilder.plot_feature_importance(best_model[-1], feature_names)
    except Exception as e:
        print(f"⚠️ 特征重要性可视化失败: {str(e)}")


if __name__ == "__main__":
    main()