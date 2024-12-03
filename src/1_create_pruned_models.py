import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.utils.prune as prune
from torch import nn
import os
from logger import setup_logger
from tqdm import tqdm


def prune_model(
    model_name: str,
    output_dir: str,
    logger,
    pruning_ratio=0.1 
):
    """
    Prune a Hugging Face model with the specified pruning ratio.
    """
    
    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Count initial parameters
    initial_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Initial parameter count: {initial_params:,}")

    # Apply pruning to all linear and embedding layers
    modules = list(model.named_modules())
    for name, module in tqdm(modules, desc='Pruning layers', total=len(modules)):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            logger.debug(f"Pruning layer: {name}")

        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            prune.remove(module, 'weight')
        
        if isinstance(module, nn.Embedding):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            prune.remove(module, 'weight')

    # remaining_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    remaining_params = sum(p.nonzero().size(0) for p in model.parameters() if p.requires_grad)

    logger.info(f"Remaining parameter count: {remaining_params:,}")
    logger.info(f"Pruned {(1 - remaining_params/initial_params)*100:.2f}% of parameters")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Pruned model saved to: {output_dir}")

    return model, tokenizer

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prune a Hugging Face model using PyTorch pruning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model',type=str,required=True,help='Name of the model on HF')
    parser.add_argument('--pruning_ratio',type=float,default=0.1,help='Percentage of weights to prune (0.0 to 1.0)')
    parser.add_argument('--output_dir',type=str,default='pruned_model',help='Directory to save the pruned model')
    parser.add_argument('--test_prompt',type=str,default='Hello, how are you?',help='Test prompt to evaluate the pruned model')
    parser.add_argument('--log_dir',type=str,default='logs',help='Directory to store log files')
        
    return parser.parse_args()


def main():
    args = parse_arguments()
    logger = setup_logger(log_dir=args.output_dir, log_file="pruning.log")
    try:
        model_name = args.model
        pruning_ratio = args.pruning_ratio
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Starting model pruning process")
        pruned_model, tokenizer = prune_model(
            model_name=model_name,
            pruning_ratio=pruning_ratio,
            output_dir=output_dir,
            logger=logger
        )
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()