import os
import time

import torch.cuda.nccl as nccl
import torch.distributed as dist

try:
    import packaging.version
except ImportError:
    from pkg_resources import packaging

from pretraining.policies import *

from torch.distributed.fsdp import ShardingStrategy


def train(
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
    n_tok,
):
    model.train()
    ddp_loss = torch.zeros(2).to(local_rank)

    start = time.time()
    loop_start = time.time()
    for batch_idx, (input, label) in enumerate(train_loader, start=start_step+1):
        if batch_idx == cfg.num_steps:
            break
        input = input.to(local_rank)
        label = label.to(local_rank)

        optimizer.zero_grad()
        output = model(input)
        ce_loss = torch.nn.CrossEntropyLoss()
        loss = ce_loss(output.view(-1, output.size(-1)), label.view(-1).long())

        loss.backward()
        optimizer.step()
        scheduler.step()

        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1

        if profiler:
            profiler.step()

        if batch_idx % cfg.report_interval == 0:
            time_now = time.time()
            elapsed_time = time_now - loop_start
            dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
            train_accuracy = ddp_loss[0] / ddp_loss[1]
            world_size = int(os.environ["WORLD_SIZE"])
            elapsed_tokens = (batch_idx - start_step) * world_size * cfg.batch_size * cfg.seq_length // cfg.tp_size
            if rank == 0:
                print("step:", batch_idx)
                print("tokens seen:", n_tok + elapsed_tokens)
                print("loss:", train_accuracy.item())
                print(f"speed for these {cfg.report_interval} steps:", (time_now - start) / cfg.report_interval)
                print("overall speed:", elapsed_time / (batch_idx - start_step))
                print("reserved memory:", torch.cuda.max_memory_reserved(device=torch.cuda.current_device()))
                print("active memory:", torch.cuda.max_memory_allocated(device=torch.cuda.current_device()))
                print("overall token per gpu per sec:", int(elapsed_tokens / world_size / elapsed_time))
                print("token per day:", int(elapsed_tokens / elapsed_time * 3600 * 24))
            start = time.time()
            ddp_loss.zero_()
        torch.cuda.reset_peak_memory_stats(device=torch.cuda.current_device())

        if batch_idx % cfg.checkpoint_interval == 0:
            overwritten = checkpointer.save(
                batch_idx,
                model,
                optimizer,
                train_loader,
                tokens_seen = elapsed_tokens + n_tok
            )

    return train_accuracy


def setup():
    """Initialize the process group for distributed training"""
    # https://github.com/michael-sandoval/hybrid_quantum_hpc/blob/main/frontier_qml/cnn_qml_distributed_mpi_1GpuPerTask.py#L337
    n_gpus_total = torch.cuda.device_count()
    print(f'Total GPUs on the system: {n_gpus_total}')

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    world_rank = rank = comm.Get_rank()
    backend = None
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(world_rank)
    os.environ['LOCAL_RANK'] = "0"

    master_addr = os.environ["MASTER_ADDR"]
    os.environ['MASTER_ADDR'] = master_addr #'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['NCCL_SOCKET_IFNAME'] = 'hsn0' #added
    print(f'Total GPUs being used this run: {world_size}')

    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    #dist.init_process_group("nccl")


def cleanup():
    """Clean up the process group after training"""
    dist.barrier()
    dist.destroy_process_group()

    from mpi4py import MPI
    if MPI.Is_initialized(): MPI.Finalize()


def setup_environ_flags():
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping and sharding strategy"""

    verify_bfloat_support = (
            (torch.version.hip or (torch.version.cuda and packaging.version.parse(torch.version.cuda).release >= (11, 0)))
            and torch.cuda.is_bf16_supported()
            and dist.is_nccl_available()
            and nccl.version() >= (2, 10)
    )

    # mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support
        if bf16_ready:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        else:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
    else:
        mixed_precision_policy = None

    # wrapping policy
    wrapping_policy = get_llama_wrapper()

    # sharding strategy
    if cfg.sharding_strategy == "fsdp":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif cfg.sharding_strategy == "hsdp":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    if rank == 0:
        print(f"Sharding strategy = {cfg.sharding_strategy}")

    return mixed_precision_policy, wrapping_policy, sharding_strategy


def get_profiler(cfg):
    if cfg.use_profiler:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                "profile_traces"
            ),
            profile_memory=True,
            with_stack=False,
            record_shapes=True,
        )
    else:
        profiler = None
    return profiler
