import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from logger import setup_logger

logger = setup_logger('../../plots', '9_generate_pruned_quantized_finetuned_performance_graphs.log')

def main():
    sns.set(style='whitegrid')
    
    logger.info('Generating plots for 9_generate_pruned_quantized_finetuned_performance_graphs')
    input_path = '../../data/PrunedMedLLMQuantizedFinetuned_metrics.csv'
    output_path = '../../plots/9_generate_pruned_quantized_finetuned_performance_graphs.png'
    logger.info(f'Input file: {input_path}')
    
    df = pd.read_csv(input_path)

    df.set_index('Model', inplace=True)
    df = df.drop(columns=['TruthfulQA - EM'])

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.1
    num_models = len(df)
    num_metrics = len(df.columns)

    index = np.arange(num_metrics)

    # Dynamically generate colors using tab10
    color_map = plt.cm.get_cmap('tab10', num_models)
    colors = [color_map(i) for i in range(num_models)]

    for i, (model, row) in enumerate(df.iterrows()):
        ax.bar(
            index + i * bar_width,
            row.values,
            bar_width,
            label=model,
            color=colors[i % len(colors)]  # Handle cases where models exceed tab10 palette
        )
    ax.set_ylim(0, 1)
    ax.set_xlabel('Benchmark')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison by Benchmark')
    ax.set_xticks(index + bar_width * (num_models - 1) / 2)
    ax.set_xticklabels(df.columns, rotation=20, ha='right')
    ax.legend(title="Model", fontsize='small', title_fontsize='medium', ncol=2)

    plt.tight_layout()
    plt.grid(True)

    plt.savefig(output_path, format='png', dpi=300)
    logger.info(f'Plot saved in {output_path}')

if __name__ == '__main__':
    main()
