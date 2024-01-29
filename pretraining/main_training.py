import math
import os

import fire
import torch
import torch.optim as optim
from fms.models.llama import LLaMAConfig, LLaMA
from fms.utils.checkpointing import Checkpointer
from torch import distributed as dist
from torch.distributed._tensor import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.tensor.parallel.fsdp import enable_2d_with_fsdp
from torch.optim.lr_scheduler import LambdaLR

import config
import policies
from pretraining.utils.config_utils import update_config
from pretraining.utils.dataset_utils import get_data_loader, get_dummy_loader
from pretraining.utils.train_utils import setup, setup_environ_flags, get_policies, train, get_profiler, cleanup


def main(**kwargs):
    # get configs
    cfg = config.train_config()
    update_config(cfg, **kwargs)

    # ensure reproducibility
    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # some setups
    setup_environ_flags()
    setup()

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if rank == 0:
        print(f"--> running with these configs {cfg}")

    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()

    # get policy
    mixed_precision_policy, wrapping_policy, sharding_strategy_policy = get_policies(cfg, rank)

    # get fms model (7B)
    llama_config = LLaMAConfig(
        src_vocab_size=32000,
        emb_dim=4096,
        norm_eps=1e-05,
        nheads=32,
        nlayers=32,
        hidden_grow_factor=11008/4096,
        multiple_of=1,
        activation_fn="silu",
        max_expected_seq_len=2048,
    )

    # https://github.com/MEllis-github/foundation-model-stack/blob/7e6f5f5204b1d01b830881ddbde168eaff24310b/fms/models/llama.py#L462
    # https://huggingface.co/TheBloke/Llama-2-70B-fp16/blob/main/config.json
    # llama_config = LLaMAConfig(
    #     src_vocab_size=50304,
    #     emb_dim=8192,
    #     norm_eps=1e-05,
    #     nheads=64,
    #     nlayers=80,
    #     kvheads=8,  
    #     hidden_grow_factor=3.5,
    #     multiple_of=1,
    #     activation_fn="swish",
    #     max_expected_seq_len=4096,
    # )

    if cfg.low_cpu_fsdp:
        with torch.device("meta"):
            model = LLaMA(llama_config, orig_init=True) #, attn_algorithm="math") # try forcing attention algorithm
        # if rank == 0:
        #     print("loading model...")
        #     model = LLaMA(llama_config, orig_init=True)
        # else:
        #     with torch.device("meta"):
        #         model = LLaMA(llama_config, orig_init=True)
    else:
        model = LLaMA(llama_config, orig_init=True)
    
    print("model loaded")

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> model has {total_params / 1e6} Million params\n")

    # get data loader
    if rank == 0:
        print("Constructing datasets...")
    if not cfg.use_dummy_dataset:
        train_loader = get_data_loader(cfg, rank, world_size)
    else:
        train_loader = get_dummy_loader(cfg, rank, world_size)
    if rank == 0:
        print("Datasets constructed!")

    # TP
    if cfg.tp_size > 1:
        assert enable_2d_with_fsdp()
        twod_mesh = init_device_mesh("cuda", (world_size // cfg.tp_size, cfg.tp_size))
        blocks = model.get_submodule("layers")
        for i, block in enumerate(blocks):
            if rank == 0:
                print("parallelization of block:", i)
            block = parallelize_module(
                module=block,
                device_mesh=twod_mesh,
                parallelize_plan={
                    "attn.query": ColwiseParallel(),
                    "attn.key": ColwiseParallel(),
                    "attn.value": ColwiseParallel(),
                    "attn.dense": RowwiseParallel(),
                    "ff_sub_layer.w1": ColwiseParallel(),
                    "ff_sub_layer.wg": ColwiseParallel(),
                    "ff_sub_layer.w2": RowwiseParallel(),
                },
                tp_mesh_dim=1,
            )
        fsdp_pg = twod_mesh.get_dim_groups()[0]
    else:
        fsdp_pg = None

    # FSDP
    model = FSDP(
        model,
        process_group=fsdp_pg,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=sharding_strategy_policy,
        use_orig_params=cfg.use_torch_compile,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        sync_module_states=cfg.low_cpu_fsdp,
        param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
        if cfg.low_cpu_fsdp else None,
    )
        # param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
        # if cfg.low_cpu_fsdp and rank != 0 else None,
        # device_mesh=init_device_mesh("cuda", (world_size // cfg.sharding_group_size, cfg.sharding_group_size))
        # if cfg.sharding_strategy == "hsdp" else None,

    # fsdp activation checkpointing
    if cfg.fsdp_activation_checkpointing:
        if rank == 0:
            print(f"--> applying FSDP activation checkpointing...")
        policies.apply_fsdp_checkpointing(model, cfg.selective_checkpointing)

    if cfg.use_torch_compile:
        print("compile not supported yet for llama ")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(.9,.95), weight_decay=0.1)

    # optionally load from checkpoint
    start_step = 0
    checkpointer = Checkpointer(cfg.ckpt_save_path, 1000, cfg.sharding_strategy, rank, local_rank)
    model, optimizer, train_loader, start_step, tokens_seen = checkpointer.load(
        model,
        optimizer,
        train_loader,
        path=cfg.ckpt_load_path
    )

    # LR schedule
    warmup_interval = min(2000, cfg.num_steps//20)
    schedule = lambda x: min(
        1 - (1 - min(x, warmup_interval) / warmup_interval) ** 2,  # parabolic anneal
        0.1 + 0.5 * (1 - 0.1) * (1 + math.cos(min(x, cfg.num_steps) / cfg.num_steps * math.pi)),
    )
    scheduler = LambdaLR(optimizer, lambda x: schedule(x + start_step))

    # profiler
    profiler = get_profiler(cfg)

    # Train
    if rank == 0:
        print(f"Training for {cfg.num_steps} steps")
    train(
        cfg,
        model,
        local_rank,
        rank,
        train_loader,
        optimizer,
        scheduler,
        profiler,
        checkpointer,
        start_step,
        tokens_seen,
    )

    cleanup()


if __name__ == "__main__":
    fire.Fire(main)
