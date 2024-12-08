import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from logger import setup_logger

logger = setup_logger('../../plots', '11_generate_mean_inference_time.log')


def main():
    sns.set(style='whitegrid')
    
    logger.info('Generating plots for 11_generate_mean_inference_time')
    input_path = '../../data/PrunedMedLLMQuantizedFinetuned_efficiency_metrics.csv'
    output_path = '../../plots/11_generate_mean_inference_time.png'
    logger.info(f'Input file: {input_path}')
    
    df = pd.read_csv(input_path)
    
    inference_columns = [
        "MedQA", "MedMCQA", "PubMedQA", "MMLU"
    ]
    
    df['Average Inference Time'] = df[inference_columns].mean(axis=1)
    
    df = df.sort_values('Average Inference Time', ascending=False)
    
    plt.figure(figsize=(12, 6))
    cmap = plt.cm.Blues
    colors = cmap(np.linspace(1, 0.2, len(df)))
    plt.barh(
        df['Model'], 
        df['Average Inference Time'], 
        color=colors
    )
    
    plt.title('Average Inference Time per Model', fontsize=14)
    plt.xlabel('Average Inference Time (s)', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    plt.grid(True)
    
    plt.savefig(output_path, dpi=300)
    logger.info(f'Plot saved in {output_path}')

if __name__ == '__main__':
    main()
