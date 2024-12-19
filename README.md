# Pruning LLMs with Lookahead
The code is builds on the [wanda](https://github.com/locuslab/wanda) codebase (see `original_README.md`). Note that we refer to lookahead pruning as easpa in this codebase.

## Setup
Installation instructions can be found in [INSTALL.md](INSTALL.md).

## Usage
Below is an example command for pruning LLaMA-2-7B with aespa, to achieve unstructured 50% sparsity. For OPT models, simply use `main_opt.py` instead.
```sh
python main.py \
    --model meta-llama/llama-7b-hf \
    --prune_method aespa \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_2_7b/unstructured/aespa/ 
```
We provide a quick overview of the arguments:  
- `--model`: The identifier for the LLaMA model on the Hugging Face model hub.
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--prune_method`: The method to prune with (e.g., aespa, wanda, sparsegpt, etc.)
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--sparsity_type`: Specifies the type of sparsity [`unstructured`, `2:4`, `4:8`].
- `--save`: Specifies the directory where the result will be stored.
