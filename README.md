# Hierarchical Reasoning Model

![](./assets/hrm.png)

Reasoning, the process of devising and executing complex goal-oriented action sequences, remains a critical challenge in AI.
Current large language models (LLMs) primarily employ Chain-of-Thought (CoT) techniques, which suffer from brittle task decomposition, extensive data requirements, and high latency. Inspired by the hierarchical and multi-timescale processing in the human brain, we propose the Hierarchical Reasoning Model (HRM), a novel recurrent architecture that attains significant computational depth while maintaining both training stability and efficiency.
HRM executes sequential reasoning tasks in a single forward pass without explicit supervision of the intermediate process, through two interdependent recurrent modules: a high-level module responsible for slow, abstract planning, and a low-level module handling rapid, detailed computations. With only 27 million parameters, HRM achieves exceptional performance on complex reasoning tasks using only 1000 training samples. The model operates without pre-training or CoT data, yet achieves nearly perfect performance on challenging tasks including symbolic expression simplification (Lisp).
Furthermore, HRM outperforms much larger models with significantly longer context windows on the Abstraction and Reasoning Corpus (ARC), a key benchmark for measuring artificial general intelligence capabilities.
These results underscore HRM’s potential as a transformative advancement toward universal computation and general-purpose reasoning systems.

 
## Quick Start Guide 🚀

### Prerequisites ⚙️

Ensure PyTorch with CUDA is installed following the official selector for your system:

```bash
# Visit the official PyTorch install page and choose your env:
# https://pytorch.org/get-started/locally/
# Example (CUDA 12.x):
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Additional build tools
pip3 install packaging ninja wheel setuptools setuptools-scm
```

FlashAttention is optional. If you install it and enable it in config, the model can use it:

```bash
# Hopper / FA-3 (see repo for details)
pip3 install flash-attn --no-build-isolation  # or build from source per docs

# Ampere or earlier (FA-2)
pip3 install flash-attn
```

Note: The Attention module respects the `use_flash_attn` flag. If FlashAttention is not installed, it will fall back to PyTorch scaled dot-product attention.

## Install Python Dependencies 🐍

```bash
pip install -r requirements.txt
```

## W&B Integration 📈

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking and metric visualization. Ensure you're logged in:

```bash
wandb login
```

## Run Experiments

### Quick Demo: Lisp Simplification 💻

Generate a symbolic expression simplification dataset and train HRM on it.

```bash
# Build Lisp dataset (defaults to data/lisp-simplification)
python dataset/build_lisp_dataset.py --output-dir data/lisp-simplification

# Start training (single GPU)
OMP_NUM_THREADS=8 python pretrain.py --config-name cfg_pretrain_lisp
```

Tip: For multi-GPU, use `torchrun --nproc-per-node 8 pretrain.py --config-name cfg_pretrain_lisp`.

## Trained Checkpoints 🚧

 - [ARC-AGI-2](https://huggingface.co/sapientinc/HRM-checkpoint-ARC-2)

To use the checkpoints, see Evaluation section below.

## Full-scale Experiments 🔵

Experiments below assume an 8-GPU setup.

### Dataset Preparation

```bash
# Initialize submodules
git submodule update --init --recursive

# ARC-1
python dataset/build_arc_dataset.py  # ARC offical + ConceptARC, 960 examples
# ARC-2
python dataset/build_arc_dataset.py --dataset-dirs dataset/raw-data/ARC-AGI-2/data --output-dir data/arc-2-aug-1000  # ARC-2 official, 1120 examples

# Lisp simplification
python dataset/build_lisp_dataset.py --output-dir data/lisp-simplification
```

### Dataset Visualization

Explore the Lisp dataset visually:

* Open `puzzle_visualizer.html` (Lisp Dataset Visualizer) in your browser.
* Upload the generated dataset folder located in `data/lisp-simplification`.

## Launch experiments

### Small-sample (1K)

ARC-1:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py 
```

*Runtime:* ~24 hours

ARC-2:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/arc-2-aug-1000
```

*Runtime:* ~24 hours (checkpoint after 8 hours is often sufficient)

Lisp Simplification:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py --config-name cfg_pretrain_lisp
```

## Evaluation

Evaluate your trained models:

* Check `eval/exact_accuracy` in W&B.
* To export predictions and metrics from a checkpoint:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 evaluate.py checkpoint=<CHECKPOINT_PATH>
```

For the Lisp dataset, you can open `lisp_eval.ipynb` to interactively inspect saved outputs.

### Additional Offline Metrics

After running evaluation with `save_outputs` that include `inputs`, `labels`, and `logits`, compute richer text-level metrics offline from saved predictions:

```bash
python tools/analyze_predictions.py --checkpoint_dir $(dirname <CHECKPOINT_PATH>)

The training script's evaluate step computes simple, GPU-friendly aggregates and saves the requested tensors. Advanced metrics are calculated here offline for modularity and clarity.
```

You may also adjust what is saved during evaluation via CLI:

```bash
torchrun --nproc-per-node 8 evaluate.py checkpoint=<CHECKPOINT_PATH> save_outputs='["inputs","labels","logits"]'
```

## Mini Lisp Interpreter (Beginner-friendly)

A small Lisp interpreter is included for learning and experimentation.

- **Module**: `utils/mini_lisp.py`
- **Run REPL**:

```bash
python utils/mini_lisp.py
```

- **Run a Lisp file**:

```bash
python utils/mini_lisp.py path/to/program.lisp
```

### Language features
- **Data**: numbers, strings, booleans `#t`/`#f`, symbols, lists
- **Special forms**: `quote` (and `'`), `if`, `define`, `set!`, `lambda`, `begin`, `and`, `or`
- **Builtins**: `+ - * / < <= > >= = eq? equal? cons car cdr list length null? list? symbol? number? boolean? pair? not display newline`
- **Utilities**: `load` to evaluate a file from within the REPL

### Examples

```lisp
; arithmetic and comparison
(+ 1 2 3)           ; 6
(if (> 3 2) 42 0)   ; 42

; lists
(define xs (list 1 2 3))
(car xs)            ; 1
(cdr xs)            ; (2 3)

; functions
(define (square x) (* x x))
(square 5)          ; 25

; lambda
((lambda (a b) (+ a b)) 3 4) ; 7

; booleans and logic
(and #t (> 3 2))    ; #t
(or #f 0 "hi")     ; 0 (truthy)
```

## Notes

 - Small-sample learning typically exhibits accuracy variance of around ±2 points.
 

## Citation 📜

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```
