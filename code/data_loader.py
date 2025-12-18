"""
数据读取和预处理
统一的数据读取模块，解决编码问题
"""
import pandas as pd
import numpy as np

def load_data(filepath='../Ａ题附件1.csv'):
    """
    读取数据并设置正确的列名
    """
    # 使用gb18030编码读取
    df = pd.read_csv(filepath, encoding='gb18030')

    # 手动设置列名（因为编码问题可能导致列名乱码）
    df.columns = ['年', '用水总量（亿立方米）', '人口（千万人）', '国内生产总值（千亿元）',
                  '农业用水（亿立方米）', '工业用水（亿立方米）',
                  '生活用水（亿立方米）', '生态用水（亿立方米）']

    return df

def load_beijing_data(filepath='../Ａ题附件2.csv'):
    """
    读取北京市数据
    """
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    return df

if __name__ == '__main__':
    # 测试数据读取
    df = load_data()
    print("=" * 80)
    print("全国数据读取成功！")
    print("=" * 80)
    print("\n数据形状:", df.shape)
    print("\n列名:", df.columns.tolist())
    print("\n前5行:")
    print(df.head())
    print("\n数据类型:")
    print(df.dtypes)
    print("\n描述性统计:")
    print(df.describe())

    # 保存处理后的数据
    df.to_csv('data_cleaned.csv', index=False, encoding='utf-8-sig')
    print("\n清洗后的数据已保存到: data_cleaned.csv")
