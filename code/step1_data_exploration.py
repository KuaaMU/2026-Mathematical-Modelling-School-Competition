"""
数据探索和预处理脚本
- 解决CSV编码问题
- 数据清洗和探索性分析
- 为问题1和问题2准备数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)

# 设置工作目录
BASE_DIR = Path(r'F:\2026数学建模选拔赛题')
DATA_DIR = BASE_DIR / 'datasets'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

def load_national_data():
    """读取全国水资源数据 (2000-2024)"""
    df = pd.read_csv(DATA_DIR / 'A题附件1.csv', encoding='gbk')

    # 重命名列（统一命名）
    df.columns = ['年份', '总用水量', '人口', 'GDP', '农业用水', '工业用水', '生活用水', '生态用水']

    # 数据类型转换
    for col in df.columns:
        if col != '年份':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def load_beijing_data():
    """读取北京市数据 (2001-2024)"""
    df = pd.read_csv(DATA_DIR / 'A题附件2.csv', encoding='gbk')
    return df

def explore_and_clean():
    """数据探索和清洗"""
    print("正在读取数据...")
    df_national = load_national_data()

    # 保存数据描述到文件
    with open(OUTPUT_DIR / 'data_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("全国水资源数据概览 (2000-2024)\n")
        f.write("=" * 80 + "\n\n")

        f.write("数据形状: {}\n\n".format(df_national.shape))
        f.write("列名: {}\n\n".format(df_national.columns.tolist()))

        f.write("数据基本统计:\n")
        f.write(str(df_national.describe()) + "\n\n")

        f.write("缺失值统计:\n")
        f.write(str(df_national.isnull().sum()) + "\n\n")

        f.write("数据前10行:\n")
        f.write(str(df_national.head(10)) + "\n\n")

    print(f"[OK] 数据概览已保存至: {OUTPUT_DIR / 'data_summary.txt'}")

    # 保存清洗后的数据
    df_national.to_csv(OUTPUT_DIR / 'national_data_clean.csv', index=False, encoding='utf-8-sig')
    print(f"[OK] 清洗后数据已保存至: {OUTPUT_DIR / 'national_data_clean.csv'}")

    return df_national

def visualize_data(df):
    """生成数据可视化"""
    print("\n正在生成可视化图表...")

    # 1. 总用水量时间序列图
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['年份'], df['总用水量'], marker='o', linewidth=2, markersize=6, color='#2E86AB')
    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('总用水量 (亿立方米)', fontsize=12)
    ax.set_title('中国总用水量变化趋势 (2000-2024)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_total_water_trend.png', dpi=300, bbox_inches='tight')
    print(f"  [OK] 图1: 总用水量趋势图")
    plt.close()

    # 2. 各类用水量对比
    fig, ax = plt.subplots(figsize=(14, 7))
    categories = ['农业用水', '工业用水', '生活用水', '生态用水']
    colors = ['#A23B72', '#F18F01', '#C73E1D', '#6A994E']

    for cat, color in zip(categories, colors):
        ax.plot(df['年份'], df[cat], marker='o', label=cat, linewidth=2, color=color)

    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('用水量 (亿立方米)', fontsize=12)
    ax.set_title('各类用水量变化对比 (2000-2024)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_category_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  [OK] 图2: 各类用水量对比图")
    plt.close()

    # 3. 相关性热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df[['总用水量', '人口', 'GDP', '农业用水', '工业用水', '生活用水', '生态用水']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    ax.set_title('水资源各指标相关性热力图', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"  [OK] 图3: 相关性热力图")
    plt.close()

    # 4. GDP与用水量关系
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # GDP趋势
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(df['年份'], df['总用水量'], marker='o', color='#2E86AB', label='总用水量')
    line2 = ax1_twin.plot(df['年份'], df['GDP'], marker='s', color='#F18F01', label='GDP')
    ax1.set_xlabel('年份', fontsize=11)
    ax1.set_ylabel('总用水量 (亿立方米)', fontsize=11, color='#2E86AB')
    ax1_twin.set_ylabel('GDP (千亿元)', fontsize=11, color='#F18F01')
    ax1.tick_params(axis='y', labelcolor='#2E86AB')
    ax1_twin.tick_params(axis='y', labelcolor='#F18F01')
    ax1.set_title('GDP与总用水量双轴对比', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 散点图
    ax2.scatter(df['GDP'], df['总用水量'], s=100, alpha=0.6, c=df['年份'], cmap='viridis')
    ax2.set_xlabel('GDP (千亿元)', fontsize=11)
    ax2.set_ylabel('总用水量 (亿立方米)', fontsize=11)
    ax2.set_title('GDP与总用水量散点关系', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_gdp_water_relationship.png', dpi=300, bbox_inches='tight')
    print(f"  [OK] 图4: GDP与用水量关系图")
    plt.close()

    # 5. 用水结构变化（堆叠面积图）
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.stackplot(df['年份'],
                 df['农业用水'], df['工业用水'], df['生活用水'], df['生态用水'],
                 labels=['农业用水', '工业用水', '生活用水', '生态用水'],
                 colors=['#A23B72', '#F18F01', '#C73E1D', '#6A994E'],
                 alpha=0.8)
    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('用水量 (亿立方米)', fontsize=12)
    ax.set_title('用水结构变化 (2000-2024)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_water_structure.png', dpi=300, bbox_inches='tight')
    print(f"  [OK] 图5: 用水结构堆叠图")
    plt.close()

def analyze_trends(df):
    """趋势分析"""
    print("\n正在进行趋势分析...")

    analysis = []
    analysis.append("=" * 80)
    analysis.append("数据趋势分析")
    analysis.append("=" * 80)
    analysis.append("")

    # 总用水量变化
    total_change = df['总用水量'].iloc[-1] - df['总用水量'].iloc[0]
    total_pct = (total_change / df['总用水量'].iloc[0]) * 100
    analysis.append(f"1. 总用水量变化:")
    analysis.append(f"   2000年: {df['总用水量'].iloc[0]:.2f} 亿立方米")
    analysis.append(f"   2024年: {df['总用水量'].iloc[-1]:.2f} 亿立方米")
    analysis.append(f"   变化: {total_change:+.2f} 亿立方米 ({total_pct:+.2f}%)")
    analysis.append("")

    # 各类用水变化
    categories = ['农业用水', '工业用水', '生活用水', '生态用水']
    analysis.append("2. 各类用水量变化:")
    for cat in categories:
        change = df[cat].iloc[-1] - df[cat].iloc[0]
        pct = (change / df[cat].iloc[0]) * 100
        analysis.append(f"   {cat}: {df[cat].iloc[0]:.2f} → {df[cat].iloc[-1]:.2f} ({pct:+.2f}%)")
    analysis.append("")

    # GDP与人口变化
    gdp_growth = ((df['GDP'].iloc[-1] / df['GDP'].iloc[0]) - 1) * 100
    pop_growth = ((df['人口'].iloc[-1] / df['人口'].iloc[0]) - 1) * 100
    analysis.append("3. 社会经济指标:")
    analysis.append(f"   GDP增长: {gdp_growth:.2f}%")
    analysis.append(f"   人口增长: {pop_growth:.2f}%")
    analysis.append(f"   人均用水量 (2000): {df['总用水量'].iloc[0]/df['人口'].iloc[0]*1000:.2f} 立方米/千人")
    analysis.append(f"   人均用水量 (2024): {df['总用水量'].iloc[-1]/df['人口'].iloc[-1]*1000:.2f} 立方米/千人")
    analysis.append("")

    # 相关性分析
    corr_gdp = df['总用水量'].corr(df['GDP'])
    corr_pop = df['总用水量'].corr(df['人口'])
    analysis.append("4. 相关性分析:")
    analysis.append(f"   总用水量 vs GDP: {corr_gdp:.4f}")
    analysis.append(f"   总用水量 vs 人口: {corr_pop:.4f}")
    analysis.append("")

    # 保存分析结果
    with open(OUTPUT_DIR / 'trend_analysis.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(analysis))

    print(f"[OK] 趋势分析已保存至: {OUTPUT_DIR / 'trend_analysis.txt'}")

    return '\n'.join(analysis)

if __name__ == "__main__":
    print("=" * 80)
    print("步骤1: 数据探索与预处理")
    print("=" * 80)

    # 1. 读取和清洗数据
    df_national = explore_and_clean()

    # 2. 生成可视化
    visualize_data(df_national)

    # 3. 趋势分析
    analysis_text = analyze_trends(df_national)

    print("\n" + "=" * 80)
    print("数据探索完成！所有结果已保存至 output/ 目录")
    print("=" * 80)
    print("\n生成的文件:")
    print("  - national_data_clean.csv     : 清洗后的数据")
    print("  - data_summary.txt            : 数据概览")
    print("  - trend_analysis.txt          : 趋势分析")
    print("  - fig1_total_water_trend.png  : 总用水量趋势")
    print("  - fig2_category_comparison.png: 各类用水对比")
    print("  - fig3_correlation_heatmap.png: 相关性热力图")
    print("  - fig4_gdp_water_relationship.png: GDP关系图")
    print("  - fig5_water_structure.png    : 用水结构图")
