"""
Question 4: Agricultural Water Pricing Optimization
问题4：农业用水最优定价策略

This module implements multi-objective optimization for agricultural water pricing,
balancing water conservation goals with farmer economic welfare.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, differential_evolution
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')

# --- 全局绘图设置 ---
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AgriculturalPricingOptimizer:
    """
    Multi-objective optimization for agricultural water pricing
    """
    
    def __init__(self):
        self.crop_data = {}
        self.optimization_results = {}
        self.pareto_solutions = []
        
    def initialize_crop_data(self):
        """
        Initialize crop-specific data for optimization
        """
        # Based on typical Chinese agricultural data
        self.crop_data = {
            'rice': {
                'water_requirement': 400,  # m³/acre baseline
                'price_elasticity': -0.25,  # moderate elasticity
                'yield_per_acre': 450,     # kg/acre
                'price_per_kg': 2.8,       # yuan/kg
                'production_cost': 800,    # yuan/acre (excluding water)
                'farmer_income_share': 0.15, # net income as share of revenue
                'area_share': 0.35         # share of total agricultural area
            },
            'wheat': {
                'water_requirement': 300,
                'price_elasticity': -0.20,
                'yield_per_acre': 350,
                'price_per_kg': 2.4,
                'production_cost': 600,
                'farmer_income_share': 0.18,
                'area_share': 0.25
            },
            'corn': {
                'water_requirement': 350,
                'price_elasticity': -0.22,
                'yield_per_acre': 400,
                'price_per_kg': 2.2,
                'production_cost': 650,
                'farmer_income_share': 0.16,
                'area_share': 0.20
            },
            'vegetables': {
                'water_requirement': 500,
                'price_elasticity': -0.35,  # higher elasticity for cash crops
                'yield_per_acre': 2000,
                'price_per_kg': 3.5,
                'production_cost': 1200,
                'farmer_income_share': 0.25,
                'area_share': 0.15
            },
            'fruits': {
                'water_requirement': 600,
                'price_elasticity': -0.40,
                'yield_per_acre': 1500,
                'price_per_kg': 5.0,
                'production_cost': 1800,
                'farmer_income_share': 0.30,
                'area_share': 0.05
            }
        }
        
        print("Crop data initialized:")
        for crop, data in self.crop_data.items():
            print(f"  {crop}: {data['water_requirement']} m³/acre, elasticity {data['price_elasticity']}")
    
    def calculate_water_demand(self, prices):
        """
        Calculate water demand for each crop given prices
        """
        demands = {}
        
        for i, (crop, data) in enumerate(self.crop_data.items()):
            base_demand = data['water_requirement']
            elasticity = data['price_elasticity']
            price = prices[i]
            
            # Assume baseline price of 0.30 yuan/m³
            baseline_price = 0.30
            price_ratio = price / baseline_price
            
            # Demand function: Q = Q₀ * (P/P₀)^ε
            demand = base_demand * (price_ratio ** elasticity)
            demands[crop] = max(demand, base_demand * 0.3)  # Minimum 30% of baseline
            
        return demands
    
    def calculate_farmer_income_impact(self, prices):
        """
        Calculate impact on farmer income from water pricing
        """
        demands = self.calculate_water_demand(prices)
        income_impacts = {}
        
        for i, (crop, data) in enumerate(self.crop_data.items()):
            # Current water cost
            current_water_cost = demands[crop] * prices[i]
            baseline_water_cost = data['water_requirement'] * 0.30
            
            # Additional water cost
            additional_cost = current_water_cost - baseline_water_cost
            
            # Revenue per acre
            revenue_per_acre = data['yield_per_acre'] * data['price_per_kg']
            
            # Net income impact as percentage of revenue
            income_impact_pct = additional_cost / revenue_per_acre
            
            income_impacts[crop] = {
                'additional_cost': additional_cost,
                'income_impact_pct': income_impact_pct,
                'water_demand': demands[crop]
            }
            
        return income_impacts
    
    def objective_water_conservation(self, prices):
        """
        Water conservation objective (minimize total water use)
        """
        demands = self.calculate_water_demand(prices)
        
        # Calculate total water consumption weighted by area
        total_water = 0
        for crop, demand in demands.items():
            area_share = self.crop_data[crop]['area_share']
            total_water += demand * area_share
            
        return total_water
    
    def objective_farmer_welfare(self, prices):
        """
        Farmer welfare objective (minimize income impact)
        """
        income_impacts = self.calculate_farmer_income_impact(prices)
        
        # Calculate weighted average income impact
        total_impact = 0
        for crop, impact in income_impacts.items():
            area_share = self.crop_data[crop]['area_share']
            farmer_share = self.crop_data[crop]['farmer_income_share']
            
            # Weight by area and farmer income dependency
            weight = area_share * farmer_share
            total_impact += impact['income_impact_pct'] * weight
            
        return total_impact
    
    def multi_objective_function(self, prices, weight_conservation=0.6):
        """
        Combined multi-objective function
        """
        # Normalize objectives
        water_obj = self.objective_water_conservation(prices) / 400  # Normalize by baseline
        welfare_obj = self.objective_farmer_welfare(prices) * 10    # Scale up welfare impact
        
        # Weighted combination
        combined_obj = weight_conservation * water_obj + (1 - weight_conservation) * welfare_obj
        
        return combined_obj
    
    def constraint_affordability(self, prices):
        """
        Affordability constraint: water cost should not exceed 8% of farmer income
        """
        income_impacts = self.calculate_farmer_income_impact(prices)
        
        max_impact = 0
        for crop, impact in income_impacts.items():
            max_impact = max(max_impact, impact['income_impact_pct'])
        
        # Constraint: max impact should be <= 0.08 (8%)
        return 0.08 - max_impact
    
    def constraint_food_security(self, prices):
        """
        Food security constraint: basic crop production should be maintained
        """
        demands = self.calculate_water_demand(prices)
        
        # Check that staple crops (rice, wheat, corn) maintain at least 90% of baseline water
        food_crops = ['rice', 'wheat', 'corn']
        min_ratio = 1.0
        
        for crop in food_crops:
            if crop in demands:
                baseline = self.crop_data[crop]['water_requirement']
                current = demands[crop]
                ratio = current / baseline
                min_ratio = min(min_ratio, ratio)
        
        # Constraint: minimum ratio should be >= 0.90
        return min_ratio - 0.90
    
    def generate_pareto_frontier(self, n_points=50):
        """
        Generate Pareto frontier using weighted sum method
        """
        print("Generating Pareto frontier...")
        
        pareto_solutions = []
        weights = np.linspace(0.1, 0.9, n_points)
        
        # Price bounds: 0.20 to 1.00 yuan/m³
        bounds = [(0.20, 1.00) for _ in range(len(self.crop_data))]
        
        for weight in weights:
            try:
                # Optimization with constraints
                constraints = [
                    {'type': 'ineq', 'fun': self.constraint_affordability},
                    {'type': 'ineq', 'fun': self.constraint_food_security}
                ]
                
                # Initial guess
                x0 = [0.40] * len(self.crop_data)
                
                result = minimize(
                    lambda x: self.multi_objective_function(x, weight),
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000}
                )
                
                if result.success:
                    prices = result.x
                    water_obj = self.objective_water_conservation(prices)
                    welfare_obj = self.objective_farmer_welfare(prices)
                    
                    solution = {
                        'weight': weight,
                        'prices': prices,
                        'water_consumption': water_obj,
                        'income_impact': welfare_obj,
                        'water_savings_pct': (400 - water_obj) / 400 * 100,
                        'feasible': True
                    }
                    
                    pareto_solutions.append(solution)
                    
            except Exception as e:
                print(f"Optimization failed for weight {weight:.2f}: {e}")
                continue
        
        self.pareto_solutions = pareto_solutions
        print(f"Generated {len(pareto_solutions)} Pareto-optimal solutions")
        
        return pareto_solutions
    
    def select_optimal_solution(self):
        """
        Select optimal solution from Pareto frontier using knee point method
        """
        if not self.pareto_solutions:
            print("No Pareto solutions available")
            return None
        
        # Calculate knee point (maximum curvature)
        solutions = self.pareto_solutions
        
        # Normalize objectives for knee point calculation
        water_values = [s['water_consumption'] for s in solutions]
        welfare_values = [s['income_impact'] for s in solutions]
        
        water_norm = [(w - min(water_values)) / (max(water_values) - min(water_values)) 
                     for w in water_values]
        welfare_norm = [(w - min(welfare_values)) / (max(welfare_values) - min(welfare_values)) 
                       for w in welfare_values]
        
        # Find knee point (minimum distance to ideal point)
        distances = []
        for i in range(len(solutions)):
            # Distance to ideal point (0, 0) in normalized space
            distance = np.sqrt(water_norm[i]**2 + welfare_norm[i]**2)
            distances.append(distance)
        
        # Select solution with minimum distance
        knee_index = np.argmin(distances)
        optimal_solution = solutions[knee_index]
        
        print(f"\nOptimal solution selected (knee point):")
        print(f"Water savings: {optimal_solution['water_savings_pct']:.1f}%")
        print(f"Income impact: {optimal_solution['income_impact']:.3f}")
        
        return optimal_solution
    
    def create_tiered_pricing_scheme(self, optimal_solution):
        """
        Create tiered pricing scheme based on optimal solution
        """
        optimal_prices = optimal_solution['prices']
        crop_names = list(self.crop_data.keys())
        
        # Create tiered structure
        tiered_scheme = {}
        
        for i, crop in enumerate(crop_names):
            base_price = optimal_prices[i]
            
            # Define tiers based on crop type
            if crop in ['rice', 'wheat', 'corn']:  # Food crops
                tiers = {
                    'basic': {
                        'threshold': (0, 300),  # m³/acre
                        'price': base_price * 0.8,  # 20% discount for basic tier
                        'description': '基础用水量'
                    },
                    'standard': {
                        'threshold': (301, 450),
                        'price': base_price,
                        'description': '标准用水量'
                    },
                    'conservation': {
                        'threshold': (451, float('inf')),
                        'price': base_price * 1.3,  # 30% premium for excess use
                        'description': '超量用水'
                    }
                }
            else:  # Economic crops (vegetables, fruits)
                tiers = {
                    'standard': {
                        'threshold': (0, 400),
                        'price': base_price,
                        'description': '标准用水量'
                    },
                    'premium': {
                        'threshold': (401, float('inf')),
                        'price': base_price * 1.5,  # Higher premium for cash crops
                        'description': '超量用水'
                    }
                }
            
            tiered_scheme[crop] = tiers
        
        return tiered_scheme
    
    def create_comprehensive_visualization(self):
        """
        Create comprehensive visualization of optimization results
        """
        os.makedirs('code/Q4/results', exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Pareto Frontier
        ax1 = axes[0, 0]
        if self.pareto_solutions:
            water_savings = [s['water_savings_pct'] for s in self.pareto_solutions]
            income_impacts = [s['income_impact'] * 100 for s in self.pareto_solutions]  # Convert to percentage
            
            ax1.scatter(income_impacts, water_savings, alpha=0.7, s=60, color='#3498DB')
            
            # Highlight optimal solution
            optimal = self.select_optimal_solution()
            if optimal:
                ax1.scatter(optimal['income_impact'] * 100, optimal['water_savings_pct'], 
                           s=200, color='#E74C3C', marker='*', 
                           edgecolor='black', linewidth=2, label='Optimal Solution')
                
                # Add annotation
                ax1.annotate(f'Optimal\n({optimal["income_impact"]*100:.1f}%, {optimal["water_savings_pct"]:.1f}%)',
                           xy=(optimal['income_impact'] * 100, optimal['water_savings_pct']),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax1.set_xlabel('Farmer Income Impact (%)')
        ax1.set_ylabel('Water Savings (%)')
        ax1.set_title('Pareto Frontier: Water Conservation vs Farmer Welfare')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Optimal Prices by Crop
        ax2 = axes[0, 1]
        if hasattr(self, 'optimal_solution') and self.optimal_solution:
            crop_names = ['Rice', 'Wheat', 'Corn', 'Vegetables', 'Fruits']
            crop_names_cn = ['水稻', '小麦', '玉米', '蔬菜', '水果']
            prices = self.optimal_solution['prices']
            
            bars = ax2.bar(crop_names, prices, color=['#2ECC71', '#3498DB', '#9B59B6', '#E67E22', '#E74C3C'])
            
            # Add value labels on bars
            for bar, price in zip(bars, prices):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{price:.2f}', ha='center', va='bottom', fontweight='bold')
            
            # Add Chinese labels below x-axis
            for i, (bar, cn_name) in enumerate(zip(bars, crop_names_cn)):
                ax2.text(bar.get_x() + bar.get_width()/2., -0.05,
                        cn_name, ha='center', va='top', fontsize=9)
            
            ax2.set_ylabel('Water Price (yuan/m³)')
            ax2.set_title('Optimal Water Prices by Crop Type')
            ax2.tick_params(axis='x', rotation=0, labelsize=9)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_ylim(-0.1, max(prices) * 1.15)
        
        # 3. Water Demand Response
        ax3 = axes[0, 2]
        if hasattr(self, 'optimal_solution') and self.optimal_solution:
            crop_names = ['Rice', 'Wheat', 'Corn', 'Vegetables', 'Fruits']
            crop_names_cn = ['水稻', '小麦', '玉米', '蔬菜', '水果']
            baseline_demands = [self.crop_data[crop]['water_requirement'] for crop in self.crop_data.keys()]
            optimal_demands = list(self.calculate_water_demand(self.optimal_solution['prices']).values())
            
            x = np.arange(len(crop_names))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, baseline_demands, width, label='Baseline', alpha=0.8, color='#95A5A6')
            bars2 = ax3.bar(x + width/2, optimal_demands, width, label='Optimal', alpha=0.8, color='#3498DB')
            
            ax3.set_xlabel('Crop Type')
            ax3.set_ylabel('Water Demand (m³/acre)')
            ax3.set_title('Water Demand: Baseline vs Optimal Pricing')
            ax3.set_xticks(x)
            ax3.set_xticklabels(crop_names, rotation=0, fontsize=9)
            
            # Add Chinese labels below
            for i, cn_name in enumerate(crop_names_cn):
                ax3.text(i, -30, cn_name, ha='center', va='top', fontsize=8)
            
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.set_ylim(-50, max(max(baseline_demands), max(optimal_demands)) * 1.1)
        
        # 4. Income Impact by Crop
        ax4 = axes[1, 0]
        if hasattr(self, 'optimal_solution') and self.optimal_solution:
            income_impacts = self.calculate_farmer_income_impact(self.optimal_solution['prices'])
            crops = ['Rice', 'Wheat', 'Corn', 'Vegetables', 'Fruits']
            crops_cn = ['水稻', '小麦', '玉米', '蔬菜', '水果']
            impacts = [income_impacts[crop]['income_impact_pct'] * 100 for crop in income_impacts.keys()]
            
            bars = ax4.bar(crops, impacts, color=['#2ECC71', '#3498DB', '#9B59B6', '#E67E22', '#E74C3C'])
            
            # Add horizontal line at 8% (affordability threshold)
            ax4.axhline(y=8, color='red', linestyle='--', alpha=0.7, label='Affordability Limit (8%)')
            
            # Add Chinese labels below
            for i, (bar, cn_name) in enumerate(zip(bars, crops_cn)):
                ax4.text(bar.get_x() + bar.get_width()/2., -0.5,
                        cn_name, ha='center', va='top', fontsize=9)
            
            ax4.set_ylabel('Income Impact (%)')
            ax4.set_title('Farmer Income Impact by Crop Type')
            ax4.tick_params(axis='x', rotation=0, labelsize=9)
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.set_ylim(-1, max(max(impacts), 8) * 1.2)
        
        # 5. Regional Adjustment Factors
        ax5 = axes[1, 1]
        regions = ['North China\nPlain', 'Yangtze River\nBasin', 'Northwest\nArid Region', 'Northeast\nRegion', 'South China\nRegion']
        regions_cn = ['华北平原', '长江流域', '西北干旱区', '东北地区', '华南地区']
        water_scarcity = [0.8, 0.6, 1.2, 0.4, 0.5]  # Scarcity multipliers
        
        bars = ax5.bar(regions, water_scarcity, color=['#E74C3C', '#F39C12', '#8E44AD', '#27AE60', '#3498DB'])
        ax5.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, label='National Average')
        
        # Add Chinese labels as text annotations
        for i, (bar, cn_name) in enumerate(zip(bars, regions_cn)):
            ax5.text(bar.get_x() + bar.get_width()/2., -0.1, cn_name, 
                    ha='center', va='top', fontsize=9, rotation=0)
        
        ax5.set_ylabel('Price Multiplier')
        ax5.set_title('Regional Water Scarcity Adjustments')
        ax5.tick_params(axis='x', rotation=0, labelsize=8)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_ylim(-0.2, 1.4)  # Adjust y-limit to accommodate Chinese labels
        
        # 6. Implementation Timeline
        ax6 = axes[1, 2]
        phases = ['Pilot Phase\n2025-2026', 'Expansion Phase\n2027-2028', 'Full Implementation\n2029-2030']
        phases_cn = ['试点期\n2025-2026', '推广期\n2027-2028', '完善期\n2029-2030']
        price_levels = [0.7, 0.85, 1.0]  # Relative to optimal price
        
        ax6.plot(range(len(phases)), price_levels, 'o-', linewidth=3, markersize=10, color='#2ECC71')
        ax6.fill_between(range(len(phases)), price_levels, alpha=0.3, color='#2ECC71')
        
        # Set English labels for x-axis
        ax6.set_xticks(range(len(phases)))
        ax6.set_xticklabels(phases, fontsize=9)
        
        # Add Chinese labels as annotations
        for i, (cn_phase, price) in enumerate(zip(phases_cn, price_levels)):
            ax6.annotate(cn_phase, xy=(i, price + 0.05), ha='center', va='bottom', 
                        fontsize=8, color='#2ECC71', fontweight='bold')
        
        ax6.set_ylabel('Price Level (Relative to Optimal)')
        ax6.set_title('Phased Implementation Timeline')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0.6, 1.1)
        
        plt.tight_layout()
        plt.savefig('code/Q4/results/agricultural_pricing_optimization.png', 
                   dpi=300, bbox_inches='tight')
        print("✓ Comprehensive optimization visualization saved")
        
        return fig
    
    def run_optimization(self):
        """
        Run the complete agricultural pricing optimization
        """
        print("=== Agricultural Water Pricing Optimization ===")
        
        # Initialize data
        self.initialize_crop_data()
        
        # Generate Pareto frontier
        pareto_solutions = self.generate_pareto_frontier()
        
        # Select optimal solution
        self.optimal_solution = self.select_optimal_solution()
        
        if self.optimal_solution:
            # Create tiered pricing scheme
            tiered_scheme = self.create_tiered_pricing_scheme(self.optimal_solution)
            
            # Print results
            self.print_optimization_results(tiered_scheme)
            
            # Create visualizations
            self.create_comprehensive_visualization()
            
            # Save results
            self.save_optimization_results(tiered_scheme)
            
            return {
                'optimal_solution': self.optimal_solution,
                'tiered_scheme': tiered_scheme,
                'pareto_solutions': pareto_solutions
            }
        else:
            print("Optimization failed to find feasible solution")
            return None
    
    def print_optimization_results(self, tiered_scheme):
        """
        Print optimization results summary
        """
        print(f"\n=== Optimization Results Summary ===")
        
        optimal = self.optimal_solution
        print(f"Water Conservation: {optimal['water_savings_pct']:.1f}% reduction")
        print(f"Farmer Income Impact: {optimal['income_impact']*100:.1f}% of revenue")
        
        print(f"\nOptimal Prices by Crop:")
        crop_names = list(self.crop_data.keys())
        for i, crop in enumerate(crop_names):
            price = optimal['prices'][i]
            print(f"  {crop}: {price:.2f} yuan/m³")
        
        print(f"\nTiered Pricing Structure (Example - Rice):")
        rice_tiers = tiered_scheme['rice']
        for tier_name, tier_data in rice_tiers.items():
            threshold = tier_data['threshold']
            price = tier_data['price']
            desc = tier_data['description']
            if threshold[1] == float('inf'):
                print(f"  {tier_name}: >{threshold[0]} m³/acre, {price:.2f} yuan/m³ ({desc})")
            else:
                print(f"  {tier_name}: {threshold[0]}-{threshold[1]} m³/acre, {price:.2f} yuan/m³ ({desc})")
    
    def save_optimization_results(self, tiered_scheme):
        """
        Save optimization results to files
        """
        os.makedirs('code/Q4/results', exist_ok=True)
        
        # Save optimal solution
        optimal_df = pd.DataFrame({
            'Crop': list(self.crop_data.keys()),
            'Optimal_Price': self.optimal_solution['prices'],
            'Baseline_Demand': [self.crop_data[crop]['water_requirement'] 
                              for crop in self.crop_data.keys()],
            'Optimal_Demand': list(self.calculate_water_demand(self.optimal_solution['prices']).values())
        })
        
        optimal_df['Water_Savings_Pct'] = ((optimal_df['Baseline_Demand'] - optimal_df['Optimal_Demand']) 
                                         / optimal_df['Baseline_Demand'] * 100)
        
        optimal_df.to_csv('code/Q4/results/optimal_pricing_solution.csv', index=False)
        
        # Save Pareto frontier
        pareto_df = pd.DataFrame(self.pareto_solutions)
        pareto_df.to_csv('code/Q4/results/pareto_frontier.csv', index=False)
        
        # Save detailed report
        with open('code/Q4/results/agricultural_pricing_report.txt', 'w', encoding='utf-8') as f:
            f.write("=== Agricultural Water Pricing Optimization Report ===\n\n")
            
            f.write("EXECUTIVE SUMMARY:\n")
            f.write(f"Water Conservation Achieved: {self.optimal_solution['water_savings_pct']:.1f}%\n")
            f.write(f"Farmer Income Impact: {self.optimal_solution['income_impact']*100:.1f}% of revenue\n")
            f.write(f"Solution Feasibility: Meets all constraints\n\n")
            
            f.write("OPTIMAL PRICES:\n")
            crop_names = list(self.crop_data.keys())
            for i, crop in enumerate(crop_names):
                price = self.optimal_solution['prices'][i]
                f.write(f"{crop}: {price:.2f} yuan/m³\n")
            
            f.write(f"\nCONSTRAINT SATISFACTION:\n")
            f.write(f"Affordability: {self.constraint_affordability(self.optimal_solution['prices']):.3f} (>0 = satisfied)\n")
            f.write(f"Food Security: {self.constraint_food_security(self.optimal_solution['prices']):.3f} (>0 = satisfied)\n")
            
            f.write(f"\nIMPLEMENTATION RECOMMENDATIONS:\n")
            f.write("1. 分阶段实施：2025-2026试点，2027-2028推广，2029-2030完善\n")
            f.write("2. 补贴机制：对低收入农户提供每亩50元补贴\n")
            f.write("3. 技术配套：推广滴灌技术，政府补贴设备费用50%\n")
            f.write("4. 监测调整：建立用水监测系统，根据效果调整价格\n")
        
        print("✓ Optimization results saved to code/Q4/results/")


def main():
    """
    Main function for agricultural pricing optimization
    """
    optimizer = AgriculturalPricingOptimizer()
    results = optimizer.run_optimization()
    
    if results:
        print("\n=== Agricultural Pricing Optimization Complete ===")
        print("Files generated:")
        print("- agricultural_pricing_optimization.png: Comprehensive visualization")
        print("- optimal_pricing_solution.csv: Optimal solution details")
        print("- pareto_frontier.csv: All Pareto-optimal solutions")
        print("- agricultural_pricing_report.txt: Detailed analysis report")
    
    return results


if __name__ == "__main__":
    results = main()