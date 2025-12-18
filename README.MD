# 🌊 水资源预测与管理 - CUMCM 2026 选拔赛项目

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/Status-In%20Progress-orange.svg)](https://github.com/your-repo)
[![Team](https://img.shields.io/badge/Team-3%20Members-green.svg)](https://github.com/orgs/your-team)

## 📋 项目概述
**题目**：A题 水资源的预测和管理
**目标**：基于2000-2016年全国数据和2001-2016年北京市数据，建立数学模型解决：
1. 2017-2021年全国用水量短期预测
2. 用水量影响因素识别与分析
3. 水价变化对工业/居民用水的影响机制
4. 农业用水合理定价策略设计

**核心要求**：
✅ 预测精度：MAPE ≤ 3%
✅ 模型验证：所有模型必须通过敏感性分析
✅ 政策落地：定价策略需考虑农民承受能力
✅ 学术规范：数据来源必须明确标注

## 🗂️ 项目结构
```bash
F:.
│
├───code/                  # 所有代码文件
│   ├───Q1/                # 问题1：预测模型代码
│   ├───Q2/                # 问题2：影响因素分析代码
│   ├───Q3/                # 问题3：水价弹性分析代码
│   ├───Q4/                # 问题4：农业定价优化代码
│   └───figures/           # 代码生成的图表（自动同步到texfile/figures/）
│
├───datasets/              # 数据集
│   ├── A题附件1.csv       # 全国数据（2000-2016）- 重点使用
│   ├── A题附件2.csv       # 北京市数据（2001-2016）- 重点使用
│   └── history_datasets/  # 历史数据（备用）
│
├───texfile/               # LaTeX论文章节
│   ├── figures/           # 最终论文图表（手动整理）
│   ├── reference.bib      # 参考文献数据库
│   └── *.tex              # 各章节LaTeX文件
│
├───temple_v0.0/           # 模板备份（勿修改）
│
├───题目/                  # 原始题目文件
│
├── MCM-main.tex           # 主LaTeX文件（**中文已修复**）
├── mcmthesis.cls          # 论文模板类文件
├── COMPILE_GUIDE.md       # 编译指南（自动生成）
└── README.md              # 本文件
