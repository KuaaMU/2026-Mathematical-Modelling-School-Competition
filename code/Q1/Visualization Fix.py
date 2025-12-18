"""
Question 1 Visualization Fix
问题1：可视化修复与美化专用脚本
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置风格
sns.set_theme(style="whitegrid", font="Arial", font_scale=1.1)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def fix_visualization(results, years_hist, y_hist):
    """
    修复坐标轴缩放问题，并添加嵌入式表格
    """
    print("正在生成修复版的高清图表...")

    # 1. 数据清洗：把模型预测中的 0 或负数 替换为 NaN，防止拉坏坐标轴
    models = ['GM(1,1)', 'Poly_Reg', 'ARIMA', 'Ensemble']
    for m in models:
        if m in results.columns:
            results[m] = results[m].apply(lambda x: np.nan if x < 10 else x)

    # 2. 创建画布
    fig, ax = plt.subplots(figsize=(12, 7))

    # 3. 绘制历史数据 (实线)
    ax.plot(years_hist, y_hist, 'o-', color='#333333', label='Actual Data (2000-2016)',
            linewidth=2, markersize=6, zorder=10)

    # 4. 绘制各模型 (虚线，颜色淡一点)
    ax.plot(results['Year'], results['GM(1,1)'], '--', color='#66c2a5', alpha=0.6, linewidth=1.5, label='GM(1,1)')
    ax.plot(results['Year'], results['Poly_Reg'], '--', color='#fc8d62', alpha=0.6, linewidth=1.5, label='Poly Reg')
    # ARIMA 单独处理，可能开头有缺省
    ax.plot(results['Year'], results['ARIMA'], '--', color='#8da0cb', alpha=0.6, linewidth=1.5, label='ARIMA')

    # 5. 绘制组合模型 (高亮红线)
    # 分段绘制：历史拟合部分 vs 未来预测部分
    ensemble_hist = results[results['Year'] <= 2016]
    ensemble_fut = results[results['Year'] >= 2016] # 2016作为连接点

    ax.plot(ensemble_hist['Year'], ensemble_hist['Ensemble'], '-', color='#E74C3C',
            linewidth=1, alpha=0.5, label='Ensemble Fit') # 历史拟合细一点

    ax.plot(ensemble_fut['Year'], ensemble_fut['Ensemble'], 'o-', color='#E74C3C',
            linewidth=3, markersize=7, label='Ensemble Forecast (2017-2021)', zorder=20) # 预测粗一点

    # 6. 关键修复：设置 Y 轴范围 (自动聚焦)
    # 排除 0 之后计算 min/max
    valid_values = np.concatenate([y_hist, results['Ensemble'].dropna().values])
    y_min = np.min(valid_values)
    y_max = np.max(valid_values)
    margin = (y_max - y_min) * 0.2 # 上下留 20% 空间
    ax.set_ylim(y_min - margin, y_max + margin)

    # 7. 添加预测区域背景阴影
    ax.axvspan(2016.5, 2021.5, color='#E74C3C', alpha=0.05)
    ax.text(2017, y_max + margin*0.1, 'Forecast Phase', color='#E74C3C', fontweight='bold')

    # 8. 添加嵌入式数据表 (在图表右下角)
    # 准备表格数据
    forecast_data = results[results['Year'] >= 2017][['Year', 'Ensemble']].copy()
    forecast_data['Ensemble'] = forecast_data['Ensemble'].round(2)
    table_vals = forecast_data.values
    col_labels = ['Year', 'Prediction (100M m³)']

    # 绘制表格
    the_table = plt.table(cellText=table_vals,
                          colLabels=col_labels,
                          colWidths=[0.1, 0.2],
                          loc='lower right',
                          cellLoc='center',
                          bbox=[0.68, 0.05, 0.3, 0.25]) # [x, y, width, height]

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1, 1.5)

    # 9. 装饰
    ax.set_title('Short-term Forecasting of National Water Consumption (2017-2021)', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Water Consumption (100 Million m³)', fontsize=12)
    ax.set_xlabel('Year', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.9)

    plt.tight_layout()
    save_path = 'question1/results/forecast_comparison_fixed.png'
    plt.savefig(save_path, dpi=300)
    print(f"修复完成！图片已保存至: {save_path}")
    plt.show()

# =======================================================
# 重新加载数据并运行修复 (你需要把这段逻辑接在之前的代码后面)
# 或者直接把下面的代码复制到之前的脚本末尾
# =======================================================

if __name__ == "__main__":
    # 模拟重新加载之前算好的 results (为了演示，我这里重新生成一下环境)
    # 在你的实际操作中，直接调用 fix_visualization(results, years_hist, y_hist) 即可

    # 这里的代码是为了确保你可以独立运行这个修复脚本
    try:
        # 尝试读取之前保存的 csv 重新构造数据
        forecast_df = pd.read_csv('question1/results/forecast_2017_2021.csv')
        # 还需要历史数据，这里我们简单模拟一下结构以便运行，实际上你应该用内存里的变量
        # 假设你已经有了 results 变量，直接跳过这一段，运行下面那行函数调用
        pass
    except:
        pass

    # *** 关键：请在之前的脚本中，evaluate_and_plot 函数被调用的地方，改用这个函数 ***
    # fix_visualization(results, years_hist, y_hist)

    # 如果你想单独测试效果，用下面的模拟数据：
    print("生成模拟数据测试修复效果...")
    years_all = np.arange(2000, 2022)
    # 模拟 ARIMA 在 2000年 是 0 的情况
    arima_bug = np.linspace(55, 63, 22); arima_bug[0] = 0

    mock_results = pd.DataFrame({
        'Year': years_all,
        'GM(1,1)': np.linspace(54, 64, 22),
        'Poly_Reg': np.linspace(55, 62, 22) + np.random.normal(0, 0.5, 22),
        'ARIMA': arima_bug, # 模拟 BUG
        'Ensemble': np.linspace(55, 62.8, 22)
    })
    mock_y_hist = np.linspace(55, 61, 17) + np.random.normal(0, 0.2, 17)
    mock_years_hist = np.arange(2000, 2017)

    fix_visualization(mock_results, mock_years_hist, mock_y_hist)
