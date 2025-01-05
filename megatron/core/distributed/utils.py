import torch
import megatron.core.parallel_state as mpu

def grouped_all_reduce(
    tensor,
    op=torch.distributed.ReduceOp.SUM
):
    intra_group = mpu.get_intra_group_all_reduce_group()
    inter_group = mpu.get_inter_group_all_reduce_group()

    torch.distributed.all_reduce(tensor, op, group=intra_group)
    if inter_group:
        assert torch.distributed.get_rank(group=intra_group) == 0, \
            "representative of intra group is not the lowest rank."
        torch.distributed.all_reduce(tensor, op, group=inter_group)
    handle = torch.distributed.broadcast(tensor, src=mpu.get_intra_group_representative(), group=intra_group)
    return handle

def warpped_all_reduce(
    tensor,
    op=torch.distributed.ReduceOp.SUM,
    group=None,
    async_op=False
):
    from megatron.training import get_args
    args = get_args()
    if args.group_communication:
        assert not async_op, \
            "grouped all-reduce should be synchronous."
        handle = grouped_all_reduce(tensor, op)
    else:
        handle = torch.distributed.all_reduce(tensor, op, group, async_op)
    return handle