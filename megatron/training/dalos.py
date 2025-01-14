import torch
import torch.distributed
import numpy as np
import math
import time

from .global_vars import get_args
from .dynamicDP.optimizer.JointOptimizer import JointOptimizer
from .dynamicDP.optimizer.SimpleAdditiveOptimizer import SimpleAdditiveOptimizer
from .dynamicDP.optimizer.NetworkOnlyOptimizer import NetworkOnlyOptimizer
from .dynamicDP.optimizer.ComputeOnlyOptimizer import ComputeOnlyOptimizer
from .dynamicDP.utils.training import TrainingBatch

class DALOS():
    def __init__(
            self,
            optimizer_type: str,
            num_gpus: int,
            num_paras: int,
            fp16: bool,
            micro_batch_size: int,
            current_global_batch_size: int,
            compute_complexity: float,
            static_compute_power,
        ):
        self.optimizer_type = optimizer_type
        if self.optimizer_type == 'joint':
            self.optimizer = JointOptimizer(num_gpus)
        elif self.optimizer_type == 'add':
            self.optimizer = SimpleAdditiveOptimizer(num_gpus)
        elif self.optimizer_type == 'net':
            self.optimizer = NetworkOnlyOptimizer(num_gpus)
        elif self.optimizer_type == 'compute':
            self.optimizer = ComputeOnlyOptimizer(num_gpus)
        else:
            raise ValueError("Unknown DALOS Optimizer")

        if fp16:
            self.grad_size = num_paras * 2.0 / (1024 * 1024 * 1024)
        else:
            self.grad_size = num_paras * 4.0 / (1024 * 1024 * 1024)
        self.micro_batch_size = micro_batch_size
        self.current_global_batch_size = current_global_batch_size
        self.compute_complexity = compute_complexity
        self.static_compute_power = static_compute_power

    def get_batch(self):
        return TrainingBatch(self.current_global_batch_size, self.compute_complexity, self.grad_size)

    def optimize(self):
        if self.static_compute_power is None:
            # TODO: profile compute power
            pass
        else:
            compute_power = self.static_compute_power

        if self.optimizer_type in ['joint', 'add', 'net']:
            bandwidth_matrix, latency_matrix = self.profile_network()
        else:
            bandwidth_matrix, latency_matrix = None, None

        result = self.optimizer.optimize(
            compute_power,
            bandwidth_matrix,
            latency_matrix,
            self.get_batch()
        )

        if self.optimizer_type in ['joint', 'add', 'compute']:
            unit = self.current_global_batch_size // self.micro_batch_size
            data_alloc = [round(x*unit) for x in result['data_distribution']]
            self.current_global_batch_size = sum(data_alloc) * self.micro_batch_size
        else:
            data_alloc = None
        if self.optimizer_type in ['joint', 'add', 'net']:
            groups = result['communication_groups']
        else:
            groups = None
        return data_alloc, groups
    
    def solve_data_distribution(self, groups):
        assert self.optimizer_type == 'joint', \
            "heuristic group communication with static/dynamic workload allocation is only for joint optimizer"

        if self.static_compute_power is None:
            # TODO: profile compute power
            pass
        else:
            compute_power = self.static_compute_power

        result = self.optimizer._solve_data_distribution(groups, compute_power, self.get_batch())

        unit = self.current_global_batch_size // self.micro_batch_size
        data_alloc = [round(x*unit) for x in result['data_distribution']]
        self.current_global_batch_size = sum(data_alloc) * self.micro_batch_size
        return data_alloc

    def profile_network(self):
        # It seems that optimizer assume bidirectional homogeneous
        # However, profile is designed for heterogeneous cases
        with torch.no_grad():
            args = get_args()
            rank = torch.distributed.get_rank()

            def get_next_match(ranks, rank):
                # input a list of ranks
                # return a target and rotate ranks list
                idx = ranks.index(rank)
                target = ranks[-1-idx]

                ranks[:] = [ranks[0], ranks[-1]] + ranks[1:-1]
                return target

            # First profile latency
            ranks = list(range(((args.world_size+1)>>1)*2))
            ltc_local_dict = {}
            for _ in range(((args.world_size+1)>>1)*2-1):
                target = get_next_match(ranks, rank)
                send = torch.randn((1,), dtype=torch.float32, device='cuda')
                recv = torch.zeros((1,), dtype=torch.float32, device='cuda')
                if target == args.world_size:
                    send_time = 0
                    recv_time = 0
                elif rank < target:
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
                ltc_local_dict[target] = (send_time + recv_time) // 2
            # ltc_local_dict[rank] = [0, 0]
            ltc_local_dict[rank] = 0
            ltc_local = []
            for r in range(args.world_size):
                ltc_local.append(ltc_local_dict[r])
            ltc_local = torch.tensor(ltc_local, dtype=torch.int, device='cuda')
            ltc_matrix = [torch.zeros_like(ltc_local, device='cuda')] * args.world_size
            torch.distributed.all_gather(ltc_matrix, ltc_local)

            # Next profile bandwidth
            ranks = list(range(((args.world_size+1)>>1)*2))
            bdw_local_dict = {}
            for _ in range(((args.world_size+1)>>1)*2-1):
                target = get_next_match(ranks, rank)
                send = torch.randn(args.profile_tensor_size, dtype=torch.float32, device='cuda')
                recv = torch.zeros(args.profile_tensor_size, dtype=torch.float32, device='cuda')
                tensor_size = 4.0 * math.prod(args.profile_tensor_size)
                if target == args.world_size:
                    send_time = 0
                    recv_time = 0
                elif rank < target:
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
                bdw_local_dict[target] = (tensor_size/send_time*1e9 + tensor_size/recv_time*1e9) / 2
            # bdw_local_dict[rank] = [0.0, 0.0]
            bdw_local_dict[rank] = 0.0
            bdw_local = []
            for r in range(args.world_size):
                bdw_local.append(bdw_local_dict[r])
            bdw_local = torch.tensor(bdw_local, dtype=torch.float, device='cuda')
            bdw_matrix = [torch.zeros_like(bdw_local, device='cuda')] * args.world_size
            torch.distributed.all_gather(bdw_matrix, bdw_local)

            return np.stack([bdw.cpu().numpy() for bdw in bdw_matrix], axis=0), np.stack([ltc.cpu().numpy() for ltc in ltc_matrix], axis=0)