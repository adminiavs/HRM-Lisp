from typing import List
import yaml
import os

import torch
import torch.distributed as dist

import pydantic
from omegaconf import OmegaConf
from pretrain import PretrainConfig, init_train_state, evaluate, create_dataloader


class EvalConfig(pydantic.BaseModel):
    checkpoint: str
    
    save_outputs: List[str] = ["inputs", "labels", "puzzle_identifiers", "logits", "q_halt_logits", "q_continue_logits"]


def launch():
    eval_cfg = EvalConfig(**OmegaConf.to_container(OmegaConf.from_cli()))  # type: ignore
    
    RANK = 0
    WORLD_SIZE = 1
    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    try:
        with open(os.path.join(os.path.dirname(eval_cfg.checkpoint), "all_config.yaml"), "r") as f:
            config = PretrainConfig(**yaml.safe_load(f))
    except FileNotFoundError:
        raise FileNotFoundError("Missing all_config.yaml next to the checkpoint. Ensure you evaluated from a proper training run.")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to load training config: {e}")

    # Allow overriding what to save during eval and ensure we have a valid checkpoint dir
    config.eval_save_outputs = eval_cfg.save_outputs
    config.checkpoint_path = os.path.dirname(eval_cfg.checkpoint)

    # Dataloader
    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)

    # Models
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)
    # Try unwrap torch.compile
    try:
        state_dict = torch.load(eval_cfg.checkpoint, map_location="cuda")
        train_state.model.load_state_dict(state_dict, assign=True)
    except Exception:
        # Try removing TorchDynamo's _orig_mod prefix
        try:
            state_dict = torch.load(eval_cfg.checkpoint, map_location="cuda")
            train_state.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}, assign=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found: {eval_cfg.checkpoint}")
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    train_state.step = 0
    ckpt_filename = os.path.basename(eval_cfg.checkpoint)
    if ckpt_filename.startswith("step_"):
        train_state.step = int(ckpt_filename.removeprefix("step_"))

    # Evaluate
    print ("Starting evaluation")
    
    train_state.model.eval()
    metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)

    if metrics is not None:
        print (metrics)


if __name__ == "__main__":
    launch()
