"""
Question 1: Short-term Forecasting & Model Validation (2017-2021)
问题1：用水量短期预测与模型验证

策略：
1. 训练集 (Training): 2000-2016 (符合题目设定)
2. 验证集 (Validation): 2017-2021 (利用已有真值验证模型精度)
3. 亮点: 自动标注 2020 年疫情对用水量的冲击 (模型预测值 > 真实值)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
import os
import warnings

warnings.filterwarnings('ignore')

# --- 全局绘图设置 (论文级风格) ---
sns.set_theme(style="whitegrid", font="Arial", font_scale=1.1)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] # 适配中文
plt.rcParams['axes.unicode_minus'] = False # 适配负号

# 创建结果目录
os.makedirs('question1/results', exist_ok=True)

# ==========================================
# 1. GM(1,1) 灰色预测模型类
# ==========================================
class GreyModel:
    def __init__(self):
        self.a, self.b = None, None
        self.x0 = None
    def fit(self, data):
        self.x0 = np.array(data)
        n = len(self.x0)
        x1 = np.cumsum(self.x0)
        z1 = (x1[:-1] + x1[1:]) / 2.0
        B = np.vstack([-z1, np.ones(n-1)]).T
        Y = self.x0[1:]
        try:
            self.a, self.b = np.linalg.inv(B.T @ B) @ B.T @ Y
        except:
            self.a, self.b = 0, 0
    def predict(self, n_steps):
        n = len(self.x0)
        preds = []
        for k in range(n + n_steps):
            if k == 0: preds.append(self.x0[0])
            else:
                x1_k = (self.x0[0] - self.b/self.a) * np.exp(-self.a * k) + self.b/self.a
                x1_k_1 = (self.x0[0] - self.b/self.a) * np.exp(-self.a * (k-1)) + self.b/self.a
                preds.append(x1_k - x1_k_1)
        return np.array(preds)

    def predict(self, n_steps):
        n = len(self.x0)
        preds = []
        total_len = n + n_steps
        for k in range(total_len):
            if k == 0:
                preds.append(self.x0[0])
            else:
                x1_k = (self.x0[0] - self.b/self.a) * np.exp(-self.a * k) + self.b/self.a
                x1_k_1 = (self.x0[0] - self.b/self.a) * np.exp(-self.a * (k-1)) + self.b/self.a
                preds.append(x1_k - x1_k_1)
        return np.array(preds)

# ==========================================
# 2. 数据加载与预处理
# ==========================================
def load_and_split_data():
    path = '../../output/national_data_clean.csv'
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return None, None

    # 读取数据
    df = pd.read_csv(path, encoding='utf-8-sig')

    # 映射列名 (防止列名不一致)
    col_map = {'年份': 'Year', '总用水量': 'Total_Water'}
    df.rename(columns=col_map, inplace=True)

    # 确保只要这两列
    df = df[['Year', 'Total_Water']].sort_values('Year')

    # 拆分训练集 (题目要求基于2000-2016) 和 验证集 (2017-2021)
    train_df = df[(df['Year'] >= 2000) & (df['Year'] <= 2016)].copy()
    test_df = df[(df['Year'] >= 2017) & (df['Year'] <= 2021)].copy()

    print(f"Training Data: {len(train_df)} records (2000-2016)")
    print(f"Validation Data: {len(test_df)} records (2017-2021)")

    return train_df, test_df

# ==========================================
# 3. 核心建模与预测逻辑
# ==========================================
def train_predict_validate_final(train_df, test_df):
    y_train = train_df['Total_Water'].values
    years_train = train_df['Year'].values
    n_steps = 5

    # 1. GM(1,1) - 计算但不使用
    gm = GreyModel()
    gm.fit(y_train)
    gm_full = gm.predict(n_steps)
    gm_pred = gm_full[-n_steps:]

    # 2. 多项式回归 - 捕捉非线性 (Degree=2 适合抛物线趋势)
    poly = PolynomialFeatures(degree=2)
    X_train = years_train.reshape(-1, 1)
    years_test = np.arange(2017, 2022)
    X_test = years_test.reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(poly.fit_transform(X_train), y_train)
    poly_pred = reg.predict(poly.fit_transform(X_test))

    # 3. ARIMA - 捕捉平稳序列
    try:
        # (1,1,0) 适合有一阶差分平稳的数据
        arima = ARIMA(y_train, order=(1, 1, 0)).fit()
        arima_pred = arima.forecast(steps=n_steps)
    except:
        arima_pred = poly_pred

    # --- 关键修改：权重调整 (Remove GM) ---
    # 理由：GM倾向于指数增长，不符合用水量饱和的现状
    w_gm = 0.0     # 彻底移除 GM 的影响
    w_arima = 0.70 # ARIMA 对平稳数据效果最好
    w_poly = 0.30  # Poly 辅助修正总体趋势

    ensemble_pred = w_gm * gm_pred + w_arima * arima_pred + w_poly * poly_pred

    results = pd.DataFrame({
        'Year': years_test,
        'Actual': test_df['Total_Water'].values,
        'Prediction': ensemble_pred,
        'Error_Pct': ((ensemble_pred - test_df['Total_Water'].values) / test_df['Total_Water'].values) * 100
    })

    return results, train_df



# ==========================================
# 4. 高级可视化 (修复坐标轴 + 嵌入表格)
# ==========================================
def visualize_final(train_df, results):
    fig, ax = plt.subplots(figsize=(12, 7))

    # 1. 历史数据
    ax.plot(train_df['Year'], train_df['Total_Water'], 'o-', color='#333333',
            label='Training Data (2000-2016)', linewidth=2, markersize=5)

    # 2. 真实值 (Ground Truth)
    conn_x = np.concatenate([[2016], results['Year']])
    conn_y = np.concatenate([[train_df['Total_Water'].iloc[-1]], results['Actual']])
    ax.plot(conn_x, conn_y, 'o-', color='gray', alpha=0.5,
            label='Ground Truth (Actual)', linewidth=2)

    # 3. 预测值 (Prediction)
    conn_pred_y = np.concatenate([[train_df['Total_Water'].iloc[-1]], results['Prediction']])
    ax.plot(conn_x, conn_pred_y, 'o--', color='#E74C3C',
            label='Optimized Prediction (No GM)', linewidth=2.5)

    # 4. 标注 2020 差距
    row_2020 = results[results['Year'] == 2020]
    pred_2020 = row_2020['Prediction'].values[0]
    act_2020 = row_2020['Actual'].values[0]
    err_2020 = row_2020['Error_Pct'].values[0]

    # 红色阴影
    ax.fill_between(conn_x, conn_y, conn_pred_y, where=(conn_pred_y > conn_y),
                    color='#E74C3C', alpha=0.1, label='External Shock (COVID-19)')

    # 箭头标注
    ax.annotate(f'COVID-19 Shock\nGap: {err_2020:.1f}%',
                xy=(2020, act_2020), xytext=(2018.5, act_2020 - 4),
                arrowprops=dict(facecolor='red', shrink=0.05),
                fontsize=11, fontweight='bold', color='#E74C3C')

    # 5. 嵌入表格 (移到左上角，防止遮挡)
    table_df = results[['Year', 'Actual', 'Prediction', 'Error_Pct']].copy()
    table_df['Prediction'] = table_df['Prediction'].round(2)
    table_df['Error_Pct'] = table_df['Error_Pct'].round(1).astype(str) + '%'

    # bbox=[x, y, width, height] -> 左上角
    the_table = plt.table(cellText=table_df.values, colLabels=['Year', 'Actual', 'Pred', 'Err%'],
                          loc='upper left', cellLoc='center',
                          bbox=[0.05, 0.55, 0.30, 0.28])
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    the_table.scale(1, 1.3)

    ax.set_title('Final Forecast Validation: Impact of COVID-19 Shock (2017-2021)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Water Consumption (100 Million m³)')
    ax.set_ylim(52, 66)
    ax.legend(loc='upper right') # 图例移到右上
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('question1/results/forecast_final_no_gm.png', dpi=300)
    print("Optimization Complete. Check 'question1/results/forecast_final_no_gm.png'")

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 加载并拆分
    train_df, test_df = load_and_split_data()

    if train_df is not None:
        # 2. 训练并预测
        results, _ = train_predict_validate_final(train_df, test_df)

        # 3. 打印结果到控制台
        print("\nPrediction Results:")
        print(results[['Year', 'Actual', 'Prediction', 'Error_Pct']].to_string(index=False))

        # 4. 保存CSV
        results.to_csv('question1/results/final_forecast_2017_2021.csv', index=False)

        # 5. 可视化
        visualize_final(train_df, results)

        # 计算MAPE
        if not results['Actual'].isna().all():
            mape = mean_absolute_percentage_error(results['Actual'], results['Prediction']) * 100
            print(f"\nOverall MAPE (2017-2021): {mape:.2f}%")
