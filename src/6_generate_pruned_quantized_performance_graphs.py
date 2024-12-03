import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from logger import setup_logger

logger = setup_logger('../../plots', '6_generate_pruned_quantized_performance_graphs.log')

def main():
    sns.set(style='whitegrid')
    
    logger.info('Generating plots for 6_generate_pruned_quantized_performance_graphs')
    input_path = '../../data/PrunedMedLLMQuantized_metrics.csv'
    output_path = '../../plots/6_generate_pruned_quantized_performance_graphs.png'
    logger.info(f'Input file: {input_path}')
    
    df = pd.read_csv(input_path)

    df.set_index('Model', inplace=True)
    # df = df.drop(columns=['Mean win rate'])
    df = df.drop(columns=['TruthfulQA - EM'])

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.15
    num_models = len(df)
    num_metrics = len(df.columns)

    index = np.arange(num_metrics)

    colors = [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd'
    ]

    for i, (model, row) in enumerate(df.iterrows()):
        ax.bar(
            index + i * bar_width,
            row.values,
            bar_width,
            label=model,
            color=colors[i]
        )
    ax.set_ylim(0, 1)
    ax.set_xlabel('Benchmark')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison by Benchmark')
    ax.set_xticks(index + bar_width * (num_models - 1) / 2)
    ax.set_xticklabels(df.columns, rotation=20, ha='right')
    ax.legend(title="Model")

    plt.tight_layout()
    plt.grid(True)

    plt.savefig(output_path, format='png', dpi=300)
    logger.info(f'Plot saved in {output_path}')

if __name__ == '__main__':
    main()
