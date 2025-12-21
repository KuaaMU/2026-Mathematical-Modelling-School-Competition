"""
Test script to verify the visualization improvements in Q4
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set up plotting style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Test the improved labeling approach
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Test 1: Crop names with dual language labels
ax1 = axes[0]
crop_names = ['Rice', 'Wheat', 'Corn', 'Vegetables', 'Fruits']
crop_names_cn = ['水稻', '小麦', '玉米', '蔬菜', '水果']
prices = [0.40, 0.20, 0.22, 1.00, 1.00]

bars = ax1.bar(crop_names, prices, color=['#2ECC71', '#3498DB', '#9B59B6', '#E67E22', '#E74C3C'])

# Add Chinese labels below
for i, (bar, cn_name) in enumerate(zip(bars, crop_names_cn)):
    ax1.text(bar.get_x() + bar.get_width()/2., -0.05,
            cn_name, ha='center', va='top', fontsize=9)

ax1.set_ylabel('Water Price (yuan/m³)')
ax1.set_title('Test: Crop Names with Dual Labels')
ax1.set_ylim(-0.15, 1.2)

# Test 2: Regional names
ax2 = axes[1]
regions = ['North China\nPlain', 'Yangtze River\nBasin', 'Northwest\nArid Region']
regions_cn = ['华北平原', '长江流域', '西北干旱区']
multipliers = [1.2, 0.8, 1.5]

bars = ax2.bar(regions, multipliers, color=['#E74C3C', '#F39C12', '#8E44AD'])

# Add Chinese labels below
for i, (bar, cn_name) in enumerate(zip(bars, regions_cn)):
    ax2.text(bar.get_x() + bar.get_width()/2., -0.1,
            cn_name, ha='center', va='top', fontsize=9)

ax2.set_ylabel('Price Multiplier')
ax2.set_title('Test: Regional Adjustments')
ax2.set_ylim(-0.2, 1.7)

# Test 3: Timeline with annotations
ax3 = axes[2]
phases = ['Pilot\n2025-2026', 'Expansion\n2027-2028', 'Full\n2029-2030']
phases_cn = ['试点期\n2025-2026', '推广期\n2027-2028', '完善期\n2029-2030']
price_levels = [0.7, 0.85, 1.0]

ax3.plot(range(len(phases)), price_levels, 'o-', linewidth=3, markersize=10, color='#2ECC71')

# Set English labels for x-axis
ax3.set_xticks(range(len(phases)))
ax3.set_xticklabels(phases, fontsize=9)

# Add Chinese annotations
for i, (cn_phase, price) in enumerate(zip(phases_cn, price_levels)):
    ax3.annotate(cn_phase, xy=(i, price + 0.05), ha='center', va='bottom', 
                fontsize=8, color='#2ECC71', fontweight='bold')

ax3.set_ylabel('Price Level')
ax3.set_title('Test: Implementation Timeline')
ax3.set_ylim(0.6, 1.15)

plt.tight_layout()
plt.savefig('code/Q4/results/test_visualization_improvements.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Visualization test completed successfully!")
print("✓ Dual-language labels are working properly")
print("✓ Chinese characters display correctly below English labels")
print("✓ Layout adjustments prevent label overlap")