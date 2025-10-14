#!/usr/bin/env python3
"""
ClipView Performance Visualization Tool
Creates comprehensive charts and plots to analyze ClipView segmentation performance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_performance_data():
    """Parse the performance data from your results"""
    
    # Your performance data
    data = [
        {"file": "airport_food", "precision": 0.889, "recall": 0.615, "f1": 0.727, "iou": 0.568},
        {"file": "ai_backpropagation", "precision": 1.000, "recall": 0.643, "f1": 0.783, "iou": 0.728},
        {"file": "ai_dangerous", "precision": 1.000, "recall": 0.571, "f1": 0.727, "iou": 0.584},
        {"file": "asteroids_to_worry", "precision": 0.667, "recall": 0.857, "f1": 0.750, "iou": 0.541},
        {"file": "bitcoin_explained", "precision": 1.000, "recall": 0.444, "f1": 0.615, "iou": 0.477},
        {"file": "breakfast_importance", "precision": 0.600, "recall": 0.750, "f1": 0.667, "iou": 0.595},
        {"file": "capitalism", "precision": 0.222, "recall": 0.500, "f1": 0.308, "iou": 0.706},
        {"file": "deep_learning_works", "precision": 1.000, "recall": 0.692, "f1": 0.818, "iou": 0.677},
        {"file": "differential_equations", "precision": 0.500, "recall": 0.091, "f1": 0.154, "iou": 0.559},
        {"file": "engineering_map", "precision": 0.800, "recall": 0.364, "f1": 0.500, "iou": 0.447},
        {"file": "fungi_map", "precision": 0.875, "recall": 0.500, "f1": 0.636, "iou": 0.561},
        {"file": "human_mistakes", "precision": 0.900, "recall": 0.529, "f1": 0.667, "iou": 0.546},
        {"file": "leetcode_patterns", "precision": 0.250, "recall": 0.111, "f1": 0.154, "iou": 0.327},
        {"file": "linux_file_system", "precision": 0.333, "recall": 0.045, "f1": 0.080, "iou": 0.421},
        {"file": "most_people_killed", "precision": 0.875, "recall": 1.000, "f1": 0.933, "iou": 0.612},
        {"file": "notpetya", "precision": 0.300, "recall": 0.600, "f1": 0.400, "iou": 0.635},
        {"file": "pantone_colors", "precision": 0.818, "recall": 0.750, "f1": 0.783, "iou": 0.640},
        {"file": "quantum_computing_map", "precision": 0.900, "recall": 0.900, "f1": 0.900, "iou": 0.642},
        {"file": "rl_essential_concepts", "precision": 0.000, "recall": 0.000, "f1": 0.000, "iou": 0.000},
        {"file": "rl_math_details", "precision": 0.500, "recall": 0.200, "f1": 0.286, "iou": 0.472},
        {"file": "schwarz_lantern", "precision": 0.750, "recall": 0.250, "f1": 0.375, "iou": 0.755},
        {"file": "seafood", "precision": 0.727, "recall": 0.800, "f1": 0.762, "iou": 0.658},
        {"file": "software_bugs", "precision": 0.889, "recall": 0.320, "f1": 0.471, "iou": 0.511},
        {"file": "traffic_jam", "precision": 0.625, "recall": 0.833, "f1": 0.714, "iou": 0.518},
        {"file": "us_minds", "precision": 0.800, "recall": 0.381, "f1": 0.516, "iou": 0.582},
        {"file": "water_intake", "precision": 0.000, "recall": 0.000, "f1": 0.000, "iou": 0.000},
    ]
    
    return pd.DataFrame(data)

def create_performance_overview(df, save_dir="visualizations"):
    """Create comprehensive performance overview charts"""
    
    Path(save_dir).mkdir(exist_ok=True)
    
    # 1. Overall Performance Bar Chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ClipView Performance Analysis Overview', fontsize=16, fontweight='bold')
    
    # Precision vs Recall scatter
    colors = ['red' if f1 < 0.3 else 'orange' if f1 < 0.6 else 'green' for f1 in df['f1']]
    scatter = ax1.scatter(df['recall'], df['precision'], c=colors, s=100, alpha=0.7)
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision vs Recall (Color = F1 Score)')
    ax1.grid(True, alpha=0.3)
    
    # Add diagonal line for F1 contours
    x = np.linspace(0, 1, 100)
    for f1_val in [0.2, 0.4, 0.6, 0.8]:
        y = f1_val * x / (2 * x - f1_val)
        y = np.where(y > 0, y, np.nan)
        ax1.plot(x, y, '--', alpha=0.3, label=f'F1={f1_val}')
    ax1.legend()
    
    # F1 Score distribution
    df_sorted = df.sort_values('f1', ascending=True)
    bars = ax2.barh(range(len(df_sorted)), df_sorted['f1'], 
                    color=['red' if x < 0.3 else 'orange' if x < 0.6 else 'green' for x in df_sorted['f1']])
    ax2.set_yticks(range(len(df_sorted)))
    ax2.set_yticklabels(df_sorted['file'], fontsize=8)
    ax2.set_xlabel('F1 Score')
    ax2.set_title('F1 Score by Video (Sorted)')
    ax2.grid(True, alpha=0.3)
    
    # IoU distribution
    ax3.hist(df['iou'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(df['iou'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["iou"].mean():.3f}')
    ax3.set_xlabel('IoU Score')
    ax3.set_ylabel('Number of Videos')
    ax3.set_title('IoU Score Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Performance categories pie chart
    excellent = (df['f1'] >= 0.8).sum()
    good = ((df['f1'] >= 0.6) & (df['f1'] < 0.8)).sum()
    fair = ((df['f1'] >= 0.3) & (df['f1'] < 0.6)).sum()
    poor = (df['f1'] < 0.3).sum()
    
    ax4.pie([excellent, good, fair, poor], 
            labels=['Excellent (≥0.8)', 'Good (0.6-0.8)', 'Fair (0.3-0.6)', 'Poor (<0.3)'],
            colors=['green', 'lightgreen', 'orange', 'red'],
            autopct='%1.1f%%')
    ax4.set_title('Performance Categories')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/performance_overview.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_comparison(df, save_dir="visualizations"):
    """Create detailed metric comparison charts"""
    
    # Metrics comparison heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Heatmap of all metrics
    metrics_df = df.set_index('file')[['precision', 'recall', 'f1', 'iou']]
    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax1, cbar_kws={'label': 'Score'})
    ax1.set_title('Performance Heatmap by Video')
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Videos')
    
    # Correlation matrix
    corr_matrix = metrics_df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, ax=ax2,
                square=True, cbar_kws={'label': 'Correlation'})
    ax2.set_title('Metric Correlations')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/detailed_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_content_type_analysis(df, save_dir="visualizations"):
    """Analyze performance by content type"""
    
    # Categorize videos by content type (based on filename patterns)
    def categorize_content(filename):
        if any(word in filename for word in ['ai', 'deep_learning', 'quantum', 'rl']):
            return 'AI/Tech'
        elif any(word in filename for word in ['map', 'engineering', 'fungi']):
            return 'Educational Maps'
        elif any(word in filename for word in ['food', 'breakfast', 'water', 'pantone']):
            return 'Lifestyle/Culture'
        elif any(word in filename for word in ['traffic', 'software', 'notpetya', 'seafood']):
            return 'Down the Rabbit Hole'
        elif any(word in filename for word in ['bitcoin', 'capitalism', 'mistakes']):
            return 'Economics/History'
        else:
            return 'Other'
    
    df['content_type'] = df['file'].apply(categorize_content)
    
    # Performance by content type
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Analysis by Content Type', fontsize=16, fontweight='bold')
    
    # Box plot of F1 scores by content type
    sns.boxplot(data=df, x='content_type', y='f1', ax=ax1)
    ax1.set_title('F1 Score Distribution by Content Type')
    ax1.set_xlabel('Content Type')
    ax1.set_ylabel('F1 Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # Average metrics by content type
    avg_by_type = df.groupby('content_type')[['precision', 'recall', 'f1', 'iou']].mean()
    avg_by_type.plot(kind='bar', ax=ax2)
    ax2.set_title('Average Performance by Content Type')
    ax2.set_xlabel('Content Type')
    ax2.set_ylabel('Score')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    
    # Count of videos by content type
    type_counts = df['content_type'].value_counts()
    ax3.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
    ax3.set_title('Distribution of Video Content Types')
    
    # Precision vs Recall by content type
    for content_type in df['content_type'].unique():
        subset = df[df['content_type'] == content_type]
        ax4.scatter(subset['recall'], subset['precision'], label=content_type, s=60, alpha=0.7)
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('Precision vs Recall by Content Type')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/content_type_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def create_summary_stats(df, save_dir="visualizations"):
    """Create summary statistics and insights"""
    
    print("="*80)
    print("CLIPVIEW PERFORMANCE ANALYSIS SUMMARY")
    print("="*80)
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Videos analyzed: {len(df)}")
    print(f"   Average Precision: {df['precision'].mean():.3f} ± {df['precision'].std():.3f}")
    print(f"   Average Recall: {df['recall'].mean():.3f} ± {df['recall'].std():.3f}")
    print(f"   Average F1-Score: {df['f1'].mean():.3f} ± {df['f1'].std():.3f}")
    print(f"   Average IoU: {df['iou'].mean():.3f} ± {df['iou'].std():.3f}")
    
    # Performance categories
    excellent = (df['f1'] >= 0.8).sum()
    good = ((df['f1'] >= 0.6) & (df['f1'] < 0.8)).sum()
    fair = ((df['f1'] >= 0.3) & (df['f1'] < 0.6)).sum()
    poor = (df['f1'] < 0.3).sum()
    
    print(f"\n🎯 PERFORMANCE BREAKDOWN:")
    print(f"   Excellent (F1 ≥ 0.8): {excellent} videos ({excellent/len(df)*100:.1f}%)")
    print(f"   Good (F1 0.6-0.8): {good} videos ({good/len(df)*100:.1f}%)")
    print(f"   Fair (F1 0.3-0.6): {fair} videos ({fair/len(df)*100:.1f}%)")
    print(f"   Poor (F1 < 0.3): {poor} videos ({poor/len(df)*100:.1f}%)")
    
    # Best and worst performers
    print(f"\n🏆 TOP 5 PERFORMERS:")
    top_5 = df.nlargest(5, 'f1')
    for idx, row in top_5.iterrows():
        print(f"   {row['file']:25} F1: {row['f1']:.3f} (P: {row['precision']:.3f}, R: {row['recall']:.3f})")
    
    print(f"\n⚠️  BOTTOM 5 PERFORMERS:")
    bottom_5 = df.nsmallest(5, 'f1')
    for idx, row in bottom_5.iterrows():
        print(f"   {row['file']:25} F1: {row['f1']:.3f} (P: {row['precision']:.3f}, R: {row['recall']:.3f})")
    
    # Insights
    print(f"\n💡 KEY INSIGHTS:")
    high_precision_low_recall = df[(df['precision'] > 0.8) & (df['recall'] < 0.5)]
    if len(high_precision_low_recall) > 0:
        print(f"   • {len(high_precision_low_recall)} videos show high precision but low recall")
        print(f"     → ClipView is conservative - accurate but misses many boundaries")
    
    zero_performance = df[df['f1'] == 0]
    if len(zero_performance) > 0:
        print(f"   • {len(zero_performance)} videos have zero performance - need investigation:")
        for _, row in zero_performance.iterrows():
            print(f"     → {row['file']}")
    
    # Save summary to file
    with open(f'{save_dir}/analysis_summary.txt', 'w') as f:
        f.write("ClipView Performance Analysis Summary\\n")
        f.write("="*50 + "\\n\\n")
        f.write(f"Overall F1-Score: {df['f1'].mean():.3f}\\n")
        f.write(f"Overall Precision: {df['precision'].mean():.3f}\\n")
        f.write(f"Overall Recall: {df['recall'].mean():.3f}\\n")
        f.write(f"Overall IoU: {df['iou'].mean():.3f}\\n")
        f.write(f"\\nExcellent performers: {excellent} videos\\n")
        f.write(f"Poor performers: {poor} videos\\n")

def main():
    """Main function to generate all visualizations"""
    
    print("🎨 Creating ClipView Performance Visualizations...")
    
    # Parse data
    df = parse_performance_data()
    
    # Create visualizations
    create_performance_overview(df)
    create_detailed_comparison(df)
    df_with_types = create_content_type_analysis(df)
    create_summary_stats(df_with_types)
    
    print("\\n✅ All visualizations created!")
    print("📁 Check the 'visualizations' folder for saved charts")
    print("🎯 Key files created:")
    print("   • performance_overview.png - Main performance charts")
    print("   • detailed_comparison.png - Detailed metrics heatmaps")
    print("   • content_type_analysis.png - Performance by video type")
    print("   • analysis_summary.txt - Text summary of insights")

if __name__ == "__main__":
    main()