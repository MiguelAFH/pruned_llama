import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from tqdm import tqdm
from logger import setup_logger


def extract_pruning_rate(model_name):
    match = re.search(r'pruned-(\d\.\d)', model_name)
    return float(match.group(1)) if match else 0

def main():
    logger = setup_logger(log_dir='../../plots', log_file='3_generate_pruned_performance_graphs.log')
    sns.set(style='whitegrid')
    
    input_path = '../../data/PrunedMedLLM_metrics.csv'
    output_path = '../../plots/pruning_rate_performance_plot.png'
    # metrics = ['MedQA - EM', 'MedMCQA - EM', 'PubMedQA - EM', 'MMLU - EM', 'TruthfulQA - EM']
    metrics = ['MedQA - EM', 'MedMCQA - EM', 'PubMedQA - EM', 'MMLU - EM']
    
    logger.info(f'Creating benchmark plots for {input_path}')
    logger.info(f'Metrics: {metrics}')
    
    
    df = pd.read_csv(input_path)
    df['Pruning Rate'] = df['Model'].apply(extract_pruning_rate)


    plt.figure(figsize=(10, 6))
    for metric in tqdm(metrics, desc='Plotting metrics'):
        plt.plot(df['Pruning Rate'], df[metric], marker='o', label=metric)

    plt.ylim(0, 1)
    plt.xlabel('Pruning Rate')
    plt.ylabel('Score')
    plt.title('Model Performance vs. Pruning Rate')
    plt.legend(title='Benchmark')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, format='png', dpi=300)
    logger.info(f'Plot saved in: {output_path}')
    logger.info('Done!')

if __name__ == '__main__':
    main()
