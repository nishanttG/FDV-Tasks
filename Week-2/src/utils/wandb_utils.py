# W&B initialization, logging artifacts, plots.
# src/utils/wandb_utils.py
import os
import wandb

def init_wandb(project="Week-2", name=None, config=None):
    if os.environ.get("LOG_WANDB", "1") == "0":
        print("W&B logging disabled via LOG_WANDB=0")
        return None
    run = wandb.init(project=project, name=name, config=config, resume="allow")
    return run

def finish_wandb():
    try:
        wandb.finish()
    except Exception:
        pass
