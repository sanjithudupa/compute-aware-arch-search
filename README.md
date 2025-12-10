# Compute-Aware Hybrid Attention Architecture Search

We present a two-stage approach for constructing hybrid transformer models that selectively replace full softmax attention layers with linear attention mechanisms. Our method uses knowledge distillation as a measurement tool to identify which layers can be safely replaced without degrading model quality.

In Stage 0, we train linear attention replacements (Gated Linear Attention and RWKV7) for each transformer layer independently, measuring how well each layer's behavior can be approximated. This provides a normalized distillation loss that quantifies each layer's "replaceability." In Stage 1, we assemble hybrid models by selecting the best-performing layers for replacement and fine-tune the complete model using knowledge distillation with a decoupled top-k KL divergence objective.

We conducted experiments using Qwen3-1.7B as the base architecture, training layer replacements for GLA (layers 1-10) and RWKV7 (layers 1-28), then constructing hybrid models and distilling from Qwen3-8B. Our key finding is that layer position strongly predicts amenability to linear attention replacement: early layers (1-7) can often be replaced with minimal quality loss, while middle layers (8-14) are significantly harder to approximate and require the expressivity of softmax attention. The hybrid models achieved 3.88-5.22x throughput speedup and 22-28x time-to-first-token speedup compared to the baseline model.

## Project Structure

**`training/`** - Training scripts for Stage 0 and Stage 1
- `train.py` - Stage 0 layer-wise linear attention training
- `distill.py`, `distill_part2.py` - Stage 1 full model distillation scripts
- Trains GLA and RWKV7 replacements for individual layers, then fine-tunes hybrid models

**`evaluation/`** - Evaluation and benchmarking scripts
- `measure_throughput.py` - Measures token throughput and time-to-first-token (TTFT)
- `throughput_analysis.py` - Generates performance comparison charts
- `test_hybrid_model.py`, `test_inference.py`, `test_generation.py` - Model testing scripts
- `jetnemotron_eval_code/` - Evaluation framework for downstream tasks

**`configs/`** - Configuration files
- `hybrid_model_configs/` - JSON configs for different hybrid model configurations (control, top10_gla, top25, top50)
- `linear_attn_configs/` - Configuration files for GLA and RWKV7 linear attention mechanisms
- `hybrid_config_generator.py` - Script to generate hybrid model configs based on Stage 0 loss rankings

**`qwen3_model/`** - Modified Qwen3 model implementation
- `modeling_qwen3.py` - Core model code with support for hybrid attention (GLA/RWKV7/full attention per layer)
- Supports loading from hybrid model configuration JSON files

**`utils/`** - Utility scripts
- `download_model.py` - Downloads and sets up Qwen3 models
- `dataset_setup.py` - Dataset preparation utilities
- `activation_hooks.py` - Tools for extracting hidden states during training

**`docs/`** - Documentation
- `index.html` - Full project report/documentation

**`models/`** - Downloaded model checkpoints
- `qwen3-1.7b/` - Base student model
- `Qwen3-8B/` - Teacher model for distillation

### Key Files

- `pyproject.toml` - Project dependencies and configuration
- `layer_results.csv` - Stage 0 training results (normalized losses per layer)
- `evaluation/timing_results.csv` - Performance benchmark results

## Main Workflow

1. **Stage 0 Training**: Run `training/train.py` to train linear attention replacements for each layer
2. **Generate Configs**: Use `configs/hybrid_config_generator.py` to create hybrid model configurations
3. **Stage 1 Training**: Run `training/distill.py` to train the full hybrid model via distillation
4. **Evaluation**: Use `evaluation/measure_throughput.py` to benchmark performance and `evaluation/throughput_analysis.py` to generate charts
