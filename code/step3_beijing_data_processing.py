#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3: Beijing Data Processing and Cleaning
处理A题附件2（北京市数据）并为问题3和问题4准备数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set matplotlib to support Chinese
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Create output directory
os.makedirs('./output/beijing', exist_ok=True)

def load_and_clean_beijing_data():
    """Load and clean Beijing data from A题附件2.csv"""
    print("="*80)
    print("北京市数据清洗与预处理")
    print("="*80)

    # Read the raw CSV file
    try:
        df_raw = pd.read_csv('../datasets/A题附件2.csv', encoding='utf-8-sig')
    except UnicodeDecodeError:
        df_raw = pd.read_csv('../datasets/A题附件2.csv', encoding='gbk')

    print("\n原始数据预览：")
    print(df_raw.head(10))

    # The file has a special structure - need to transpose and restructure
    # First row contains years from 2001-2024
    years = df_raw.columns[1:].tolist()
    years = [int(year.replace('年', '')) for year in years if '年' in year]

    print(f"\n年份范围: {min(years)} - {max(years)}")

    # Create structured dataframe
    data_dict = {
        'Year': years
    }

    # Extract each indicator
    indicator_mapping = {
        '年用水量(百亿立方米)': 'Total_Water',
        '农业': 'Agricultural_Water',
        '工业': 'Industrial_Water',
        '生活': 'Domestic_Water',
        '生态': 'Ecological_Water',
        '年降水量(毫米)': 'Precipitation',
        '常住人口（万人）': 'Population'
    }

    for idx, row in df_raw.iterrows():
        indicator = row.iloc[0]
        if pd.isna(indicator):
            continue
        if indicator in indicator_mapping:
            column_name = indicator_mapping[indicator]
            values = []
            for year_col in df_raw.columns[1:]:
                if '年' in year_col:
                    val = row[year_col]
                    # Handle empty or non-numeric values
                    if pd.isna(val) or val == '' or val == ' ':
                        values.append(np.nan)
                    else:
                        try:
                            values.append(float(val))
                        except:
                            values.append(np.nan)
            data_dict[column_name] = values[:len(years)]

    # Create DataFrame
    df = pd.DataFrame(data_dict)

    # Extract water price data (only available for certain years)
    # Agricultural water price
    agri_price_data = []
    indus_price_data = []

    for idx, row in df_raw.iterrows():
        indicator = row.iloc[0]
        if pd.isna(indicator):
            continue
        indicator = str(indicator).strip()
        if '水价综合' in indicator:
            # This is agricultural comprehensive price
            values = []
            for year_col in df_raw.columns[1:]:
                if '年' in year_col:
                    val = row[year_col]
                    if pd.isna(val) or val == '' or val == ' ':
                        values.append(np.nan)
                    else:
                        try:
                            values.append(float(val))
                        except:
                            values.append(np.nan)
            agri_price_data = values[:len(years)]
        elif indicator == '水价综合' and len(indus_price_data) == 0:
            # Industrial comprehensive price
            values = []
            for year_col in df_raw.columns[1:]:
                if '年' in year_col:
                    val = row[year_col]
                    if pd.isna(val) or val == '' or val == ' ':
                        values.append(np.nan)
                    else:
                        try:
                            values.append(float(val))
                        except:
                            values.append(np.nan)
            indus_price_data = values[:len(years)]

    # Find the row indices for water prices
    # Agricultural water price row (around row 11-13)
    # Industrial water price row (around row 16-18)

    agri_base_price = []
    agri_resource_price = []
    agri_comprehensive_price = []

    indus_base_price = []
    indus_resource_price = []
    indus_comprehensive_price = []

    for idx, row in df_raw.iterrows():
        indicator = str(row.iloc[0]).strip()
        values = []
        for year_col in df_raw.columns[1:]:
            if '年' in year_col:
                val = row[year_col]
                if pd.isna(val) or val == '' or val == ' ' or val == '  ':
                    values.append(np.nan)
                else:
                    try:
                        values.append(float(val))
                    except:
                        values.append(np.nan)
        values = values[:len(years)]

        # Agricultural prices (rows 11-13)
        if idx == 10 and '基本' in indicator:  # 农业基本水价
            agri_base_price = values
        elif idx == 11 and '资源' in indicator:  # 农业资源水价
            agri_resource_price = values
        elif idx == 12 and '综合' in indicator:  # 农业水价综合
            agri_comprehensive_price = values

        # Industrial prices (rows 15-17)
        elif idx == 15 and '基本' in indicator:  # 工业基本水价
            indus_base_price = values
        elif idx == 16 and '资源' in indicator:  # 工业资源水价
            indus_resource_price = values
        elif idx == 17 and '综合' in indicator:  # 工业水价综合
            indus_comprehensive_price = values

    # Add price data to dataframe
    if agri_comprehensive_price:
        df['Agri_Price'] = agri_comprehensive_price
    if indus_comprehensive_price:
        df['Indus_Price'] = indus_comprehensive_price

    print("\n清洗后的数据：")
    print(df)

    print("\n数据统计信息：")
    print(df.describe())

    # Save cleaned data
    output_file = './output/beijing/beijing_data_clean.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n[OK] 清洗后数据已保存: {output_file}")

    # Visualize water consumption trends
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Total water consumption trend
    ax1 = axes[0, 0]
    ax1.plot(df['Year'], df['Total_Water'], marker='o', linewidth=2, color='steelblue')
    ax1.set_title('北京市总用水量趋势 (2001-2024)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('年份', fontsize=10)
    ax1.set_ylabel('用水量 (百亿立方米)', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Water consumption by category
    ax2 = axes[0, 1]
    categories = ['Agricultural_Water', 'Industrial_Water', 'Domestic_Water', 'Ecological_Water']
    category_labels = ['农业', '工业', '生活', '生态']

    for cat, label in zip(categories, category_labels):
        if cat in df.columns:
            ax2.plot(df['Year'], df[cat], marker='o', label=label, linewidth=2)

    ax2.set_title('北京市分类用水量趋势', fontsize=12, fontweight='bold')
    ax2.set_xlabel('年份', fontsize=10)
    ax2.set_ylabel('用水量 (百亿立方米)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Water price trends (if available)
    ax3 = axes[1, 0]
    if 'Agri_Price' in df.columns:
        df_price_agri = df[df['Agri_Price'].notna()]
        ax3.plot(df_price_agri['Year'], df_price_agri['Agri_Price'],
                marker='s', label='农业水价', linewidth=2, color='green')

    if 'Indus_Price' in df.columns:
        df_price_indus = df[df['Indus_Price'].notna()]
        ax3.plot(df_price_indus['Year'], df_price_indus['Indus_Price'],
                marker='^', label='工业水价', linewidth=2, color='orange')

    ax3.set_title('北京市水价变化趋势', fontsize=12, fontweight='bold')
    ax3.set_xlabel('年份', fontsize=10)
    ax3.set_ylabel('水价 (元/立方米)', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Population trend
    ax4 = axes[1, 1]
    if 'Population' in df.columns:
        ax4.plot(df['Year'], df['Population'], marker='o', linewidth=2, color='purple')
        ax4.set_title('北京市常住人口趋势', fontsize=12, fontweight='bold')
        ax4.set_xlabel('年份', fontsize=10)
        ax4.set_ylabel('人口 (万人)', fontsize=10)
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_file = './output/beijing/beijing_data_overview.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"[OK] 可视化图表已保存: {fig_file}")
    plt.close()

    return df

if __name__ == "__main__":
    print("\n" + "="*80)
    print("步骤3: 北京市数据处理")
    print("="*80 + "\n")

    # Load and clean data
    df_beijing = load_and_clean_beijing_data()

    print("\n" + "="*80)
    print("数据处理完成！")
    print("="*80)
