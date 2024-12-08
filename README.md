# Pruned Llama

Quantization and pruning techniques have demonstrated significant efficacy in
reducing the size of large language models (LLMs) while maintaining their perfor-
mance across a range of general tasks. These techniques are relevant in resource-
constrained environments, such as in medical applications, where foundation mod-
els (FMs) must operate efficiently without sacrificing performance. Despite their
success in general settings, there has been limited exploration of these techniques
in the medical domain. In this paper, we evaluate the combination of quantization
and pruning on a state-of-the-art LLM (Llama 3.2 1B) using medical knowledge
datasets such as PubMedQA and MedQA. In particular, we show that a model with
20% pruning ratio, 4-bit quantization and fine-tuned achieves a 1.76x speedup for
only a 3% average score reduction compared the base model. Additionally, we ex-
plore the models’ robustness through the TruthfulQA benchmark. This work aims
to provide insights into the feasibility of using compressed LLMs in healthcare
settings and assess their ability to retain both knowledge capacity and robustness.
