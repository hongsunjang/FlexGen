import torch
import torch.distributed as dist

_COMM_DEVICE = None
_PIPELINE_PARALLEL_PRED_GROUP = None
_PIPELINE_PARALLEL_SUCC_GROUP = None

def initialize_distributed(head_ip, port, world_size, rank, local_rank,
                           comm_device):
    if rank < 2:
        port+=1
    else:
        pass

    print(f'Initializing distributed environment at {head_ip}:{port}, '
          f'world_size={world_size}, rank={rank}, local_rank={local_rank}.')

    # Initialize distributed environment
    torch.cuda.set_device(local_rank)
    distributed_init_method = f'tcp://{head_ip}:{port}'

    global _COMM_DEVICE
    _COMM_DEVICE = comm_device
    if comm_device == 'cpu':
        backend = 'gloo'
    elif comm_device == 'gpu':
        backend = 'nccl'
    else:
        raise ValueError(f'Unknown comm_device: {comm_device}')

    if rank < 2: 
        dist.init_process_group(backend=backend,
                            init_method=distributed_init_method,
                            world_size=2,
                            rank=rank)
    else:
        dist.init_process_group(backend=backend,
                            init_method=distributed_init_method,
                            world_size=2,
                            rank=rank-2)

    # Create groups for pipeline parallelism
    global _PIPELINE_PARALLEL_PRED_GROUP, _PIPELINE_PARALLEL_SUCC_GROUP
    if rank < 2:
        for pred in range(2):
            succ = (pred + 1) % 2
            group = dist.new_group([pred, succ])
            if pred == rank:
                _PIPELINE_PARALLEL_PRED_GROUP = group
            if succ == rank:
                _PIPELINE_PARALLEL_SUCC_GROUP = group
    else:
        for pred in range(2):
            group = dist.new_group([0, 1])
            if rank == 2:
                _PIPELINE_PARALLEL_PRED_GROUP = group
            if rank == 3:
                _PIPELINE_PARALLEL_SUCC_GROUP = group


    suppress_output(rank)
    print("Finished initializing distributed environment")

    return dist

def get_pipeline_parallel_pred_group():
    return _PIPELINE_PARALLEL_PRED_GROUP

def get_pipeline_parallel_succ_group():
    return _PIPELINE_PARALLEL_SUCC_GROUP

def get_comm_device():
    return _COMM_DEVICE

def suppress_output(rank):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', True)
        if force:
            builtin_print("rank #%d:" % rank, *args, **kwargs, flush=True)
        elif rank == 0:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
