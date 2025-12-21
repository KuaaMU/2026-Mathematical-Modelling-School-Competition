#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完善北京市数据：添加总用水量和水价数据
"""

import pandas as pd
import numpy as np

# Load cleaned data
df = pd.read_csv('./output/beijing/beijing_data_clean.csv', encoding='utf-8-sig')

# Add Total_Water column (sum of all categories)
df['Total_Water'] = (df['Agricultural_Water'] + df['Industrial_Water'] +
                      df['Domestic_Water'] + df['Ecological_Water'])

# Manually add water price data based on the original CSV
# Agricultural comprehensive water price (only 2007-2015 available)
agri_price_dict = {
    2007: 3.7, 2008: 3.7, 2009: 3.7,
    2010: 4.0, 2011: 4.0, 2012: 4.0, 2013: 4.0, 2014: 4.0,
    2015: 5.0
}

# Industrial comprehensive water price (only 2007-2016 available)
indus_price_dict = {
    2007: 5.6, 2008: 5.6, 2009: 5.6,
    2010: 6.21, 2011: 6.21, 2012: 6.21, 2013: 6.21, 2014: 6.21,
    2015: 9.92, 2016: 9.92
}

# Add price columns
df['Agri_Price'] = df['Year'].map(agri_price_dict)
df['Indus_Price'] = df['Year'].map(indus_price_dict)

# Calculate per capita water consumption (万人 -> 人,百亿立方米 -> 立方米)
# Population in 万人 (10000 people), Water in 百亿立方米 (billion cubic meters)
# 1 百亿立方米 = 10^9 cubic meters, 1 万人 = 10^4 people
# Per capita = (Water * 10^9) / (Population * 10^4) = Water * 10^5 / Population
df['Per_Capita_Water'] = df['Total_Water'] * 100000 / df['Population']  # m³/person/year

print("完善后的数据：")
print(df)

print("\n数据列：", df.columns.tolist())

# Save enhanced data
df.to_csv('./output/beijing/beijing_data_enhanced.csv', index=False, encoding='utf-8-sig')
print("\n[OK] 增强数据已保存: ./output/beijing/beijing_data_enhanced.csv")
