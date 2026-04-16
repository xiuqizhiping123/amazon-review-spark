import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs('data/results', exist_ok=True)
df = pd.read_csv('data/results/metrics.csv')
def plot_model_comparison(df, category_name, ax):
    category_df = df[df['category'] == category_name]
    metrics_to_plot = ['accuracy', 'f1', 'weightedPrecision', 'weightedRecall']
    plot_data = category_df.set_index('modelName')[metrics_to_plot]
    plot_data = plot_data.rename(columns={
        'accuracy': 'Accuracy',
        'f1': 'F1',
        'weightedPrecision': 'Precision',
        'weightedRecall': 'Recall'
    })
    sns.heatmap(plot_data, annot=True, cmap='YlGnBu', fmt='.4f', vmin=0.5, vmax=1.0, ax=ax)
    ax.set_title(f'Category: {category_name}')
    ax.set_ylabel('Model Name')
    ax.set_xlabel('Metrics')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_model_comparison(df, 'All_Beauty', axes[0])
plot_model_comparison(df, 'Gift_Cards', axes[1])
plt.suptitle('Sentiment Analysis Model Performance Comparison', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
save_path = 'data/results/heatmap_comparison.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"save path: {save_path}")