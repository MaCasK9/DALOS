import torch
import torch.distributed
import numpy as np
import time
import math
import sys

if __name__ == "__main__":
    assert len(sys.argv) == 3, \
        "usage: python network.py (world size) (rank)"
    
    world_size = int(sys.argv[1])
    rank = int(sys.argv[2])
    torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    torch.cuda.set_device(0)
    if rank == 0:
        print(f"Distributed world size: {world_size} inited.")

    profile_tensor_size = (2048, 2048)

    def get_next_match(ranks, rank):
        # input a list of ranks
        # return a target and rotate ranks list
        idx = ranks.index(rank)
        target = ranks[-1-idx]

        ranks[:] = [ranks[0], ranks[-1]] + ranks[1:-1]
        return target

    dummy_tensor = torch.tensor([0], dtype=torch.float32, device='cuda')

    # First profile latency
    ranks = list(range(((world_size+1)>>1)*2))
    ltc_local_dict = {}
    for _ in range(((world_size+1)>>1)*2-1):
        target = get_next_match(ranks, rank)
        send = torch.randn((1,), dtype=torch.float32, device='cuda')
        recv = torch.zeros((1,), dtype=torch.float32, device='cuda')
        if target == world_size:
            send_time = 0
            recv_time = 0
        elif rank < target:
            torch.distributed.all_reduce(dummy_tensor)
            torch.cuda.synchronize()
            start_time = time.perf_counter_ns()
            torch.distributed.send(send, target)
            torch.cuda.synchronize()
            mid_time = time.perf_counter_ns()
            torch.distributed.recv(recv, target)
            torch.cuda.synchronize()
            end_time = time.perf_counter_ns()
            send_time = mid_time - start_time
            recv_time = end_time - mid_time
        else:
            torch.distributed.all_reduce(dummy_tensor)
            torch.cuda.synchronize()
            start_time = time.perf_counter_ns()
            torch.distributed.recv(recv, target)
            torch.cuda.synchronize()
            mid_time = time.perf_counter_ns()
            torch.distributed.send(send, target)
            torch.cuda.synchronize()
            end_time = time.perf_counter_ns()
            send_time = end_time - mid_time
            recv_time = mid_time - start_time
        # ltc_local_dict[target] = [send_time, recv_time]
        ltc_local_dict[target] = (send_time + recv_time) // 2 // 1e6
    # ltc_local_dict[rank] = [0, 0]
    ltc_local_dict[rank] = 0
    ltc_local = []
    for r in range(world_size):
        ltc_local.append(ltc_local_dict[r])
    ltc_local = torch.tensor(ltc_local, dtype=torch.int64, device='cuda')
    ltc_matrix = []
    for _ in range(world_size):
        ltc_matrix.append(torch.zeros_like(ltc_local, device='cuda'))
    torch.distributed.all_gather(ltc_matrix, ltc_local)

    # Next profile bandwidth
    ranks = list(range(((world_size+1)>>1)*2))
    bdw_local_dict = {}
    for _ in range(((world_size+1)>>1)*2-1):
        target = get_next_match(ranks, rank)
        send = torch.randn(profile_tensor_size, dtype=torch.float32, device='cuda')
        recv = torch.zeros(profile_tensor_size, dtype=torch.float32, device='cuda')
        tensor_size = 4.0 * math.prod(profile_tensor_size)
        if target == world_size:
            send_time = 0
            recv_time = 0
        elif rank < target:
            #torch.distributed.all_reduce(dummy_tensor)
            torch.cuda.synchronize()
            start_time = time.perf_counter_ns()
            torch.distributed.send(send, target)
            torch.cuda.synchronize()
            mid_time = time.perf_counter_ns()
            torch.distributed.recv(recv, target)
            torch.cuda.synchronize()
            end_time = time.perf_counter_ns()
            send_time = mid_time - start_time
            recv_time = end_time - mid_time
        else:
            #torch.distributed.all_reduce(dummy_tensor)
            torch.cuda.synchronize()
            start_time = time.perf_counter_ns()
            torch.distributed.recv(recv, target)
            torch.cuda.synchronize()
            mid_time = time.perf_counter_ns()
            torch.distributed.send(send, target)
            torch.cuda.synchronize()
            end_time = time.perf_counter_ns()
            send_time = end_time - mid_time
            recv_time = mid_time - start_time
        # bdw_local_dict[target] = [tensor_size/send_time*1e9, tensor_size/recv_time*1e9]
        bdw_local_dict[target] = (tensor_size/send_time + tensor_size/recv_time) / 2
    # bdw_local_dict[rank] = [0.0, 0.0]
    bdw_local_dict[rank] = 0.0
    bdw_local = []
    for r in range(world_size):
        bdw_local.append(bdw_local_dict[r])
    bdw_local = torch.tensor(bdw_local, dtype=torch.float, device='cuda')
    bdw_matrix = []
    for _ in range(world_size):
        bdw_matrix.append(torch.zeros_like(bdw_local, device='cuda'))
    torch.distributed.all_gather(bdw_matrix, bdw_local)

    print("Test over. Bandwidth: {}. Latency: {}.".format(
        np.stack([bdw.cpu().numpy() for bdw in bdw_matrix], axis=0),
        np.stack([ltc.cpu().numpy() for ltc in ltc_matrix], axis=0)))