import os
import json
import numpy as np
import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from argdantic import ArgParser
from pydantic import BaseModel

from common import PuzzleDatasetMetadata

cli = ArgParser()

# --- Symbolic Expression Generation and Simplification ---

class Expr:
    def __init__(self):
        pass

    def __repr__(self):
        return str(self)

@dataclass
class Const(Expr):
    val: int
    def __str__(self):
        return str(self.val)

@dataclass
class Var(Expr):
    name: str
    def __str__(self):
        return self.name

@dataclass
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr
    def __str__(self):
        return f"({self.op} {self.left} {self.right})"

def generate_expr(depth: int) -> Expr:
    if depth == 0:
        if random.random() < 0.5:
            return Const(random.randint(0, 9))
        else:
            return Var(random.choice(['x', 'y', 'z']))
    else:
        op = random.choice(['+', '*', '-', '/'])
        left = generate_expr(depth - 1)
        right = generate_expr(depth - 1)
        return BinOp(op, left, right)

def simplify(expr: Expr) -> Expr:
    if isinstance(expr, BinOp):
        left = simplify(expr.left)
        right = simplify(expr.right)
        op = expr.op

        # Constant folding
        if isinstance(left, Const) and isinstance(right, Const):
            if op == '+': return Const(left.val + right.val)
            if op == '*': return Const(left.val * right.val)
            if op == '-': return Const(left.val - right.val)
            if op == '/' and right.val != 0: return Const(left.val // right.val)

        # Identity elements
        if op == '+' and isinstance(left, Const) and left.val == 0: return right
        if op == '+' and isinstance(right, Const) and right.val == 0: return left
        if op == '*' and isinstance(left, Const) and left.val == 1: return right
        if op == '*' and isinstance(right, Const) and right.val == 1: return left
        if op == '*' and isinstance(left, Const) and left.val == 0: return Const(0)
        if op == '*' and isinstance(right, Const) and right.val == 0: return Const(0)
        if op == '-' and isinstance(right, Const) and right.val == 0: return left

        return BinOp(op, left, right)
    return expr

def to_tokens(expr: Expr) -> List[str]:
    return str(expr).replace('(', '( ').replace(')', ' )').split()

# --- Dataset Creation ---

class DataProcessConfig(BaseModel):
    output_dir: str = "data/lisp-simplification"
    num_samples: int = 10000
    max_depth: int = 3
    seq_len: int = 100
    seed: int = 42

def create_lisp_dataset(config: DataProcessConfig):
    random.seed(config.seed)
    np.random.seed(config.seed)

    print("Generating Lisp expression dataset...")
    inputs, labels = [], []
    for _ in range(config.num_samples):
        expr = generate_expr(config.max_depth)
        simplified_expr = simplify(expr)
        inputs.append(to_tokens(expr))
        labels.append(to_tokens(simplified_expr))

    # Build vocabulary
    all_tokens = set()
    for seq in inputs + labels:
        all_tokens.update(seq)
    
    vocab = {tok: i + 1 for i, tok in enumerate(sorted(list(all_tokens)))}
    vocab['<pad>'] = 0
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Vocabulary: {vocab}")

    def tokenize_and_pad(seq_list: List[List[str]]) -> np.ndarray:
        padded_seqs = []
        for seq in seq_list:
            token_ids = [vocab[tok] for tok in seq]
            if len(token_ids) > config.seq_len:
                token_ids = token_ids[:config.seq_len]
            else:
                token_ids += [vocab['<pad>']] * (config.seq_len - len(token_ids))
            padded_seqs.append(token_ids)
        return np.array(padded_seqs, dtype=np.int32)

    inputs_np = tokenize_and_pad(inputs)
    labels_np = tokenize_and_pad(labels)
    
    # Create train/test split
    split_idx = int(config.num_samples * 0.9)
    train_inputs, test_inputs = inputs_np[:split_idx], inputs_np[split_idx:]
    train_labels, test_labels = labels_np[:split_idx], labels_np[split_idx:]
    
    # Save datasets
    for split_name, (split_inputs, split_labels) in {'train': (train_inputs, train_labels), 'test': (test_inputs, test_labels)}.items():
        save_dir = os.path.join(config.output_dir, split_name)
        os.makedirs(save_dir, exist_ok=True)
        
        num_samples = len(split_inputs)
        results = {
            "inputs": split_inputs,
            "labels": split_labels,
            "puzzle_identifiers": np.zeros(num_samples, dtype=np.int32),
            "puzzle_indices": np.arange(num_samples + 1, dtype=np.int32),
            "group_indices": np.arange(num_samples + 1, dtype=np.int32)
        }

        for k, v in results.items():
            np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

        metadata = PuzzleDatasetMetadata(
            seq_len=config.seq_len,
            vocab_size=len(vocab),
            pad_id=vocab['<pad>'],
            ignore_label_id=vocab['<pad>'],
            blank_identifier_id=0,
            num_puzzle_identifiers=1,
            total_groups=num_samples,
            mean_puzzle_examples=1.0,
            sets=["all"]
        )
        with open(os.path.join(save_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)

    # Save vocabulary
    with open(os.path.join(config.output_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    # For visualizer compatibility
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        rev_vocab = {v: k for k,v in vocab.items()}
        identifiers = [rev_vocab.get(i, "") for i in range(len(vocab))]
        json.dump(identifiers, f)

    print(f"Dataset saved to {config.output_dir}")

@cli.command(singleton=True)
def main(config: DataProcessConfig):
    create_lisp_dataset(config)

if __name__ == "__main__":
    cli()