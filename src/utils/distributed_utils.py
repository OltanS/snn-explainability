import torch

def ddp_setup(local_rank):
    '''
    Initialize distributed training environment
    args:
        local_rank (int): The rank of the current process
    '''
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")

def ddp_cleanup():
    '''Destroy distributed training environment'''
    torch.distributed.destroy_process_group()

def training_optimisations(model):
    '''
    Set some optimisations for training
    args:
        model (torch.nn.Module): The model to optimise
    '''
    torch.backends.cudnn.benchmark = True
    torch.compile(model, mode="max-autotune")

def ddp_model(model, local_rank):
    '''
    Wrap model for distributed training, and perform some optimisations
    args:
        model (torch.nn.Module): The model to wrap
        local_rank (int): The rank of the current process
    '''
    model.to(local_rank)
    training_optimisations(model)
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])