#!/usr/bin/env python3
"""
Interactive ClipView Performance Dashboard
Simple, focused visualizations for quick analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def create_quick_dashboard():
    """Create a quick performance dashboard"""
    
    # Your data in a more readable format
    performance_data = {
        'quantum_computing_map': {'P': 0.900, 'R': 0.900, 'F1': 0.900, 'IoU': 0.642},
        'most_people_killed': {'P': 0.875, 'R': 1.000, 'F1': 0.933, 'IoU': 0.612},
        'deep_learning_works': {'P': 1.000, 'R': 0.692, 'F1': 0.818, 'IoU': 0.677},
        'ai_backpropagation': {'P': 1.000, 'R': 0.643, 'F1': 0.783, 'IoU': 0.728},
        'pantone_colors': {'P': 0.818, 'R': 0.750, 'F1': 0.783, 'IoU': 0.640},
        'seafood': {'P': 0.727, 'R': 0.800, 'F1': 0.762, 'IoU': 0.658},
        'asteroids_to_worry': {'P': 0.667, 'R': 0.857, 'F1': 0.750, 'IoU': 0.541},
        'ai_dangerous': {'P': 1.000, 'R': 0.571, 'F1': 0.727, 'IoU': 0.584},
        'airport_food': {'P': 0.889, 'R': 0.615, 'F1': 0.727, 'IoU': 0.568},
        'traffic_jam': {'P': 0.625, 'R': 0.833, 'F1': 0.714, 'IoU': 0.518},
        'breakfast_importance': {'P': 0.600, 'R': 0.750, 'F1': 0.667, 'IoU': 0.595},
        'human_mistakes': {'P': 0.900, 'R': 0.529, 'F1': 0.667, 'IoU': 0.546},
        'fungi_map': {'P': 0.875, 'R': 0.500, 'F1': 0.636, 'IoU': 0.561},
        'bitcoin_explained': {'P': 1.000, 'R': 0.444, 'F1': 0.615, 'IoU': 0.477},
        'us_minds': {'P': 0.800, 'R': 0.381, 'F1': 0.516, 'IoU': 0.582},
        'engineering_map': {'P': 0.800, 'R': 0.364, 'F1': 0.500, 'IoU': 0.447},
        'software_bugs': {'P': 0.889, 'R': 0.320, 'F1': 0.471, 'IoU': 0.511},
        'notpetya': {'P': 0.300, 'R': 0.600, 'F1': 0.400, 'IoU': 0.635},
        'schwarz_lantern': {'P': 0.750, 'R': 0.250, 'F1': 0.375, 'IoU': 0.755},
        'capitalism': {'P': 0.222, 'R': 0.500, 'F1': 0.308, 'IoU': 0.706},
        'rl_math_details': {'P': 0.500, 'R': 0.200, 'F1': 0.286, 'IoU': 0.472},
        'differential_equations': {'P': 0.500, 'R': 0.091, 'F1': 0.154, 'IoU': 0.559},
        'leetcode_patterns': {'P': 0.250, 'R': 0.111, 'F1': 0.154, 'IoU': 0.327},
        'linux_file_system': {'P': 0.333, 'R': 0.045, 'F1': 0.080, 'IoU': 0.421},
        'rl_essential_concepts': {'P': 0.000, 'R': 0.000, 'F1': 0.000, 'IoU': 0.000},
        'water_intake': {'P': 0.000, 'R': 0.000, 'F1': 0.000, 'IoU': 0.000},
    }
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame.from_dict(performance_data, orient='index')
    df.index.name = 'Video'
    df = df.reset_index()
    
    # Create dashboard
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('ClipView Performance Dashboard', fontsize=20, fontweight='bold')
    
    # 1. F1 Score Ranking (Top subplot)
    ax1 = plt.subplot(2, 3, (1, 2))
    df_sorted = df.sort_values('F1', ascending=True)
    colors = ['#d62728' if x < 0.3 else '#ff7f0e' if x < 0.6 else '#2ca02c' for x in df_sorted['F1']]
    bars = ax1.barh(range(len(df_sorted)), df_sorted['F1'], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels([name.replace('_', ' ').title() for name in df_sorted['Video']], fontsize=9)
    ax1.set_xlabel('F1 Score', fontsize=12)
    ax1.set_title('F1 Score by Video (Red: Poor, Orange: Fair, Green: Good)', fontsize=14)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, df_sorted['F1'])):
        if value > 0:
            ax1.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', va='center', fontsize=8)
    
    # 2. Precision vs Recall scatter
    ax2 = plt.subplot(2, 3, 3)
    scatter = ax2.scatter(df['R'], df['P'], c=df['F1'], s=100, alpha=0.7, cmap='RdYlGn')
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision vs Recall\\n(Color = F1 Score)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('F1 Score', fontsize=10)
    
    # Add some video labels for interesting points
    high_recall = df[df['R'] > 0.8]
    for _, row in high_recall.iterrows():
        ax2.annotate(row['Video'].replace('_', ' ')[:15], 
                    (row['R'], row['P']), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8, alpha=0.8)
    
    # 3. Performance Distribution
    ax3 = plt.subplot(2, 3, 4)
    metrics = ['P', 'R', 'F1', 'IoU']
    metric_names = ['Precision', 'Recall', 'F1-Score', 'IoU']
    box_data = [df[metric].values for metric in metrics]
    
    bp = ax3.boxplot(box_data, labels=metric_names, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax3.set_title('Metric Distributions', fontsize=14)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. Top/Bottom Performers
    ax4 = plt.subplot(2, 3, 5)
    top_3 = df.nlargest(3, 'F1')
    bottom_3 = df.nsmallest(3, 'F1')
    
    y_pos = np.arange(3)
    ax4.barh(y_pos, top_3['F1'], color='green', alpha=0.7, label='Top 3')
    ax4.barh(y_pos - 3.5, bottom_3['F1'], color='red', alpha=0.7, label='Bottom 3')
    
    ax4.set_yticks(list(y_pos) + list(y_pos - 3.5))
    ax4.set_yticklabels(list(top_3['Video'].str.replace('_', ' ').str.title()) + 
                       list(bottom_3['Video'].str.replace('_', ' ').str.title()), fontsize=10)
    ax4.set_xlabel('F1 Score', fontsize=12)
    ax4.set_title('Best vs Worst Performers', fontsize=14)
    ax4.legend()
    ax4.grid(axis='x', alpha=0.3)
    
    # 5. Summary Statistics
    ax5 = plt.subplot(2, 3, 6)
    ax5.axis('off')  # Turn off axis for text
    
    # Calculate summary stats
    summary_text = f"""
    📊 SUMMARY STATISTICS
    
    Videos Analyzed: {len(df)}
    
    📈 AVERAGES:
    Precision: {df['P'].mean():.3f}
    Recall: {df['R'].mean():.3f}
    F1-Score: {df['F1'].mean():.3f}
    IoU: {df['IoU'].mean():.3f}
    
    🎯 PERFORMANCE LEVELS:
    Excellent (F1≥0.8): {(df['F1'] >= 0.8).sum()} videos
    Good (F1≥0.6): {(df['F1'] >= 0.6).sum()} videos
    Fair (F1≥0.3): {(df['F1'] >= 0.3).sum()} videos
    Poor (F1<0.3): {(df['F1'] < 0.3).sum()} videos
    
    🏆 BEST PERFORMER:
    {df.loc[df['F1'].idxmax(), 'Video'].replace('_', ' ').title()}
    F1: {df['F1'].max():.3f}
    
    ⚠️ NEEDS IMPROVEMENT:
    {(df['F1'] == 0).sum()} videos with zero performance
    """
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('clipview_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print quick insights
    print("\\n" + "="*60)
    print("🎯 QUICK INSIGHTS")
    print("="*60)
    print(f"Overall Performance: {df['F1'].mean():.1%} F1-Score")
    print(f"Best Video: {df.loc[df['F1'].idxmax(), 'Video'].replace('_', ' ').title()} ({df['F1'].max():.1%})")
    print(f"High Precision, Low Recall: {((df['P'] > 0.8) & (df['R'] < 0.5)).sum()} videos")
    print(f"Complete Failures: {(df['F1'] == 0).sum()} videos")
    print("\\n💡 ClipView is conservative - when it detects boundaries, they're usually correct!")

if __name__ == "__main__":
    create_quick_dashboard()