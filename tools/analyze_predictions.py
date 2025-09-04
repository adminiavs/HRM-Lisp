#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import yaml

from utils.text_metrics import token_accuracy, levenshtein_similarity


def load_vocab(vocab_path: str) -> Dict[int, str]:
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    # Stored as token->id; invert
    inv = {int(v): k for k, v in vocab.items()}
    return inv


def ids_to_tokens(ids: List[int], inv_vocab: Dict[int, str]) -> List[str]:
    return [inv_vocab.get(int(i), "<unk>") for i in ids]


def tokens_to_str(tokens: List[str]) -> str:
    # For Lisp, tokens already include parentheses and symbols
    return " ".join(t for t in tokens if t != "<pad>")


# --- Simple AST and evaluator for semantic equivalence ---
@dataclass
class Node:
    pass


@dataclass
class Const(Node):
    val: int


@dataclass
class Var(Node):
    name: str


@dataclass
class BinOp(Node):
    op: str
    left: Node
    right: Node


def parse_tokens(tokens: List[str]) -> Optional[Node]:
    def parse_at(i: int) -> Tuple[Optional[Node], int]:
        if i >= len(tokens):
            return None, i
        tok = tokens[i]
        if tok == "(":
            # ( op expr expr )
            if i + 1 >= len(tokens):
                return None, i + 1
            op = tokens[i + 1]
            left, j = parse_at(i + 2)
            if left is None:
                return None, j
            right, k = parse_at(j)
            if right is None:
                return None, k
            # Expect ")" at k
            if k < len(tokens) and tokens[k] == ")":
                return BinOp(op, left, right), k + 1
            return None, k
        # Atom
        if tok == ")":
            return None, i + 1
        # number or variable
        try:
            return Const(int(tok)), i + 1
        except Exception:  # noqa: BLE001
            return Var(tok), i + 1

    node, _j = parse_at(0)
    return node


def eval_ast(node: Node, env: Dict[str, int]) -> Optional[int]:
    try:
        if isinstance(node, Const):
            return node.val
        if isinstance(node, Var):
            return int(env[node.name])
        if isinstance(node, BinOp):
            a = eval_ast(node.left, env)
            b = eval_ast(node.right, env)
            if a is None or b is None:
                return None
            if node.op == "+":
                return a + b
            if node.op == "-":
                return a - b
            if node.op == "*":
                return a * b
            if node.op == "/":
                if b == 0:
                    return None
                return int(a // b)
        return None
    except Exception:  # noqa: BLE001
        return None


def semantic_equivalent(tokens_a: List[str], tokens_b: List[str], trials: int = 5) -> bool:
    # Remove pads
    a = [t for t in tokens_a if t != "<pad>"]
    b = [t for t in tokens_b if t != "<pad>"]
    ast_a = parse_tokens(a)
    ast_b = parse_tokens(b)
    if ast_a is None or ast_b is None:
        return False
    vars_involved = {t for t in a + b if t.isalpha() and len(t) == 1}
    # Try random assignments
    for _ in range(trials):
        env = {v: random.randint(-5, 5) for v in vars_involved}
        va = eval_ast(ast_a, env)
        vb = eval_ast(ast_b, env)
        if va is None or vb is None:
            # skip this trial
            continue
        if va != vb:
            return False
    # If all tried were equal (or couldn't evaluate), assume equivalent only if we had at least one successful evaluation
    for _ in range(trials):
        env = {v: random.randint(-5, 5) for v in vars_involved}
        va = eval_ast(ast_a, env)
        vb = eval_ast(ast_b, env)
        if va is not None and vb is not None:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved predictions for additional metrics")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory containing saved _all_preds.* files")
    parser.add_argument("--vocab_path", default=None, help="Path to vocab.json. If omitted, tries checkpoint_dir/vocab.json or data_path/vocab.json from all_config.yaml")
    parser.add_argument("--pad_token", default="<pad>")
    parser.add_argument("--max_files", type=int, default=0, help="Optional cap on number of files to read")
    args = parser.parse_args()

    vocab_path: Optional[str] = args.vocab_path
    if vocab_path is None:
        # Try local
        candidate = os.path.join(args.checkpoint_dir, "vocab.json")
        if os.path.exists(candidate):
            vocab_path = candidate
        else:
            # Try reading data_path from all_config.yaml
            cfg_path = os.path.join(args.checkpoint_dir, "all_config.yaml")
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, "r") as f:
                        cfg = yaml.safe_load(f)
                    data_path = cfg.get("data_path", None)
                    if data_path is not None:
                        candidate = os.path.join(data_path, "vocab.json")
                        if os.path.exists(candidate):
                            vocab_path = candidate
                except Exception:  # noqa: BLE001
                    pass
    if vocab_path is None:
        raise FileNotFoundError("Could not locate vocab.json. Provide --vocab_path explicitly.")
    inv_vocab = load_vocab(vocab_path)

    files = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*_all_preds.*")))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No *_all_preds.* files found in {args.checkpoint_dir}")

    total_examples = 0
    sum_token_acc = 0.0
    sum_lev_sim = 0.0
    exact_matches = 0
    sem_equiv_matches = 0

    for fp in files:
        try:
            blob = torch.load(fp, map_location="cpu")
        except Exception as e:  # noqa: BLE001
            print(f"Warning: failed to load {fp}: {e}")
            continue

        inputs = blob.get("inputs")
        labels = blob.get("labels")
        logits = blob.get("logits")
        if inputs is None or labels is None or logits is None:
            print(f"Warning: file {fp} missing required keys; skipping")
            continue

        pred_ids = torch.argmax(logits, dim=-1)

        for i in range(pred_ids.shape[0]):
            y_true_ids = labels[i].tolist()
            y_pred_ids = pred_ids[i].tolist()
            y_true_tokens = ids_to_tokens(y_true_ids, inv_vocab)
            y_pred_tokens = ids_to_tokens(y_pred_ids, inv_vocab)

            # Metrics
            ta = token_accuracy(y_true_tokens, y_pred_tokens, pad_token=args.pad_token)
            sum_token_acc += ta

            true_str = tokens_to_str(y_true_tokens)
            pred_str = tokens_to_str(y_pred_tokens)
            sum_lev_sim += levenshtein_similarity(true_str, pred_str)
            exact_matches += int(true_str == pred_str)
            if semantic_equivalent(y_true_tokens, y_pred_tokens):
                sem_equiv_matches += 1
            total_examples += 1

    if total_examples == 0:
        raise RuntimeError("No examples aggregated from prediction files.")

    print(json.dumps({
        "examples": total_examples,
        "token_accuracy": sum_token_acc / total_examples,
        "normalized_levenshtein_similarity": sum_lev_sim / total_examples,
        "exact_match_ratio": exact_matches / total_examples,
        "semantic_equivalence_ratio": sem_equiv_matches / total_examples,
    }, indent=2))


if __name__ == "__main__":
    main()


