import os
import csv

# ================= 配置区域 =================

# 根目录名称
BASE_DIR = "data"

# 时间跨度
START_YEAR = 1997
END_YEAR = 2024

# 定义四个文件的结构 (文件名, 中文列名列表, 英文列名列表)
FILES_CONFIG = [
    {
        "filename": "1_一级水资源区_供用水量.csv",
        "header_cn": ["水资源一级区", "地表水源", "地下水源", "其他水源", "供水总量", "生活用水", "工业用水", "农业用水", "生态用水", "用水总量"],
        "header_en": ["Zone_Name", "Supply_Surface", "Supply_Ground", "Supply_Other", "Supply_Total", "Use_Living", "Use_Industry", "Use_Agri", "Use_Eco", "Use_Total"]
    },
    {
        "filename": "2_省级行政区_供用水量.csv",
        "header_cn": ["省级行政区", "地表水源", "地下水源", "其他水源", "供水总量", "生活用水", "工业用水", "农业用水", "生态用水", "用水总量"],
        "header_en": ["Province", "Supply_Surface", "Supply_Ground", "Supply_Other", "Supply_Total", "Use_Living", "Use_Industry", "Use_Agri", "Use_Eco", "Use_Total"]
    },
    {
        "filename": "3_一级水资源区_用水指标.csv",
        "header_cn": ["水资源一级区", "人均综合用水量(m3)", "万元GDP用水量(m3)", "耕地灌溉亩均用水量(m3)", "人均生活用水量(L/d)", "万元工业增加值用水量(m3)"],
        "header_en": ["Zone_Name", "PerCapita_Total_Use", "Use_Per_10k_GDP", "Irrigation_Per_Mu", "PerCapita_Living_Daily", "Use_Per_10k_Industry_VA"]
    },
    {
        "filename": "4_省级行政区_用水指标.csv",
        "header_cn": ["省级行政区", "人均综合用水量(m3)", "万元GDP用水量(m3)", "耕地灌溉亩均用水量(m3)", "农田灌溉水有效利用系数", "人均生活用水量(L/d)", "万元工业增加值用水量(m3)"],
        "header_en": ["Province", "PerCapita_Total_Use", "Use_Per_10k_GDP", "Irrigation_Per_Mu", "Irrigation_Coeff", "PerCapita_Living_Daily", "Use_Per_10k_Industry_VA"]
    }
]

# ================= 执行逻辑 =================

def create_structure():
    # 1. 创建根目录
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
        print(f"✅ 创建根目录: {BASE_DIR}")

    # 2. 遍历年份
    for year in range(START_YEAR, END_YEAR + 1):
        year_dir = os.path.join(BASE_DIR, str(year))

        # 创建年份文件夹
        if not os.path.exists(year_dir):
            os.makedirs(year_dir)

        # 3. 在该年份文件夹内创建4个CSV文件
        for file_info in FILES_CONFIG:
            file_path = os.path.join(year_dir, file_info["filename"])

            # 使用 utf-8-sig 编码，防止 Excel 打开中文乱码
            with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                # 写入第一行中文
                writer.writerow(file_info["header_cn"])
                # 写入第二行英文
                writer.writerow(file_info["header_en"])

        print(f"📂 已生成 {year} 年的文件夹及模板")

    # 4. 生成一个说明文件给队友
    readme_path = os.path.join(BASE_DIR, "队友填表必读.txt")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("【填表说明】\n")
        f.write("1. 请不要修改前两行（列名），从第三行开始录入数据。\n")
        f.write("2. 如果某一年没有该数据（例如早期没有生态用水），请留空，不要填0，也不要删除列。\n")
        f.write("3. 省份名称请统一使用简称（如：北京，不要写北京市）。\n")
        f.write("4. 单位请严格按照表头说明（通常是亿立方米），不要随意换算。\n")
        f.write("5. 2024年的数据如果还没出，文件夹先空着即可。\n")

    print(f"\n✨ 任务完成！结构已生成在 '{BASE_DIR}' 目录下。")

if __name__ == "__main__":
    create_structure()
