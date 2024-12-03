import os
import bitsandbytes

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from logger import setup_logger

logger = setup_logger('../../models', '4_create_quantized_models.log')

def load_and_quantize_model(model_name):
    quantization_config_4bit = BitsAndBytesConfig(load_in_8bit=True)
    model_4bit = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=quantization_config_4bit,
        device_map="auto"
    )
    quantization_config_8bit = BitsAndBytesConfig(load_in_4bit=True)
    model_8bit = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=quantization_config_8bit,
        device_map="auto" 
    )
    logger.info(f"Quantized model {model_name} for 4 and 8 bits")
    return model_4bit, model_8bit

def main():
    model_names =[
        'meta-llama/Llama-3.2-1B-Instruct',
        '../../models/Llama-3.2-1B-Instruct_pruned_0.1'
    ]
    base_save_path = '../../models/'
    
    for model_name in tqdm(model_names, desc='Quantizing models'):
        base_name = os.path.basename(model_name)
        model_4bit, model_8bit = load_and_quantize_model(model_name)
        save_path_4bit = os.path.join(base_save_path, f'{base_name}-4bit')
        save_path_8bit = os.path.join(base_save_path, f'{base_name}-8bit')
        model_4bit.save_pretrained(save_path_4bit)
        model_8bit.save_pretrained(save_path_8bit)
        logger.info(f'Quantized model on 4-bit saved to {save_path_4bit}')
        logger.info(f'Quantized model on 8-bit saved to {save_path_8bit}')

if __name__ == '__main__':
    main()
