from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import hivemind
from hivemind import DHT, get_dht_time
import torch
from hivemind import BatchTensorDescriptor, TensorDescriptor
from hivemind.moe.expert_uid import ExpertUID
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.utils import get_logger
from flexgen.utils import (Task, ExecutionEnv, GB, T, ValueHolder,
    array_1d, array_2d, array_3d, str2bool, project_decode_latency,
    torch_mem_stats, torch_dtype_to_np_dtype, write_benchmark_log,
    read_benchmark_log)
from tensor_parallel import TensorParallel
#from tensor_parallel.tensor_parallel import PerDeviceTensors
from transformers import PretrainedConfig

from flexgen.dist_flex_opt import DistOptLM, OptLM
from flexgen.flex_opt import (Policy, InputEmbed, OutputEmbed, SelfAttention,
                              MLP, TransformerLayer, OptLM, get_filename,
                              add_parser_arguments, get_test_inputs,
                              DUMMY_WEIGHT)
from flexgen.client_manager import RemoteSequential, RemoteSequenceManager
import msgpack
import msgpack_numpy as m
from flexgen.timer import timers

class DecLM(OptLM):
    """ Decentralized LM model with assigned layers"""
    def __init__(self, config, env, path, policy, device_rank, num_blocks, dht):
        super().__init__(config, env, path, policy)
        
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        
        self.num_blocks = num_blocks

        
        layers = []
        if device_rank == 0:
            layers.append(InputEmbed(self.config, self.env, self.policy))
        block_sizes = [config.num_hidden_layers // num_blocks
                                + int(i < config.num_hidden_layers % num_blocks)
                                for i in range(num_blocks)]
        layer_start_ids = [0]
        for block_size in block_sizes:
            layer_start_ids.append(layer_start_ids[-1] + block_size)
        for i in range(layer_start_ids[device_rank], layer_start_ids[device_rank + 1]):
            if self.policy.sep_layer:
                layers.append(SelfAttention(self.config, self.env, self.policy, i))
                layers.append(MLP(self.config, self.env, self.policy, i))
            else:
                layers.append(TransformerLayer(self.config, self.env, self.policy, i))
        if device_rank == num_blocks - 1:
            layers.append(OutputEmbed(self.config, self.env, self.policy))
        self.layers = layers # layers in the current block 
        self.num_layers = len(layers)

        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        self.task = None
        self.init_all_weights()

        self.h = RemoteSequential(config, dht=dht) # decentrial 
        self.dht = dht

    # ## connect for loop in dist_flex_opt & send receive
    # def generation_step(self):
    #     self.sending_tag = 0
    #     self.receiving_tag = 0
    #     last_sending_job = None

    #     for b in range(self.num_pipeline_batches // self.num_inner_iterations):
    #         for i in range(self.execute_gen_len):
    #             for t in range(self.num_inner_iterations):
    #                 timer_name = "generate-prompt" if i == 0 else "generate"
    #                 # timers(timer_name).start()
    #                 for k in range(self.num_gpu_batches):
    #                     self.update_attention_mask(b, t, i, k)

    #                 # if self.num_pipeline_stages > 1:
    #                 #     self.send_recv_hidden(last_sending_job, (t, i))

    #                 for j in range(self.num_layers):
    #                     for k in range(self.num_gpu_batches):
    #                         self.load_weight(b, t, i, j, k)
    #                     self.sync()

    #                     for k in range(self.num_gpu_batches):
    #                         self.load_cache(t, i, j, k)
    #                         self.load_hidden(b, t, i, j, k)
    #                         self.sync()
    #                         self.compute_layer(t, i, j, k)
    #                         self.sync()
    #                         self.store_hidden(b, t, i, j, k)
    #                         self.store_cache(t, i, j, k)
    #                         self.sync()

    # def send_hidden(self, t, i, j, k, tag=0, async_=False):
    #     # Suppose we need to send tensors on GPUs
    #     x = self.hidden[t][i][j][k]
    #     val = x.pop().move(self.comm_device)
    #     receiver_rank = (self.device_rank + 1) % self.num_blocks
    #     # future = self.dht.store((t,i,j,k), msgpack.packb(val.data.numpy(), default=m.encode), expiration_time=1)
    #     # return future
    #     # if async_:
    #     #     future = dist.isend(val.data, receiver_rank, tag=tag)
    #     #     return future
    #     # else:
    #     #     dist.send(val.data, receiver_rank, tag=tag) 

    #     # Share your model and optimizer on the DHT
    #     # self.dht.store('model', val.data, tags=['model'], expiration_time=1) # expiration_time
    #     print('val.data.shape', val.data.size())
    #     print('val.data.device', val.data.device)
    #     print('type of val.data',type(val.data))
    #     key = msgpack.packb((t,i,j,k), default=m.encode)
    #     # key = '('+str(t)+','+str(i)+','+str(j)+','+str(k)+')'
    #     value_ = msgpack.packb(val.data.cpu().numpy(), default=m.encode)
    #     future = self.dht.store(key, value_, expiration_time=get_dht_time()+600)
    #     print('return send_hidden future', future)
    #     return future

    # def recv_hidden(self, t, i, j, k, tag=0, async_=False):
    #     sender_rank = (self.device_rank - 1) % self.num_blocks
    #     val_holder = self.hidden[t][i][j][k]
    #     seq_len = self.task.prompt_len if i == 0 else 1
    #     shape, dtype = self.layers[j].input_act_shape_and_dtype(
    #         self.policy.gpu_batch_size, seq_len)
    #     if val_holder.val is None:
    #         val_holder.val = self.comm_device.allocate(shape, dtype)
    #     else:
    #         val_holder.val = val_holder.val.move(self.comm_device)
    #     def move_value_callback():
    #         val_holder.val = val_holder.val.move(self.act_home)
    #     # if async_:
    #     #     future = dist.irecv(val_holder.val.data, sender_rank, tag=tag)
    #     #     return future, move_value_callback
    #     # else:
    #     #     dist.recv(val_holder.val.data, sender_rank, tag=tag)
        
    #     # future = self.dht.get('model', latest=True)
        
    #     print("For incoming connections, use:", self.dht.get_visible_maddrs())
        
            
    #     key = msgpack.packb((t,i,j,k), default=m.encode)
    #     # key = '('+str(t)+','+str(i)+','+str(j)+','+str(k)+')'
    #     print('key ', key)
    #     print('self.dht ', self.dht)
    #     future = self.dht.get(key, latest=True)
    #     print('self.dht ', self.dht)
    #     print('self.dht.get((t,i,j,k), latest=True) ', self.dht.get(key, latest=True))
    #     print('return recv_hidden future', future)
    #     return future, move_value_callback
    
    # def generation_loop_overlap_one_batch(self):
    #     assert self.num_gpu_batches == 1
    #     # Prologue
    #     self.load_weight(0, 0, 0, 0, 0)
    #     self.sync()
    #     self.sending_tag = 0
    #     self.receiving_tag = 0
    #     last_sending_job = None

    #     # Generate
    #     for b in range(self.num_pipeline_batches // self.num_inner_iterations):
    #         for i in range(self.execute_gen_len):
    #             for t in range(self.num_inner_iterations):
    #                 timer_name = "generate-prompt" if i == 0 else "generate"
    #                 timers(timer_name).start()
    #                 self.update_attention_mask(b, t, i, 0)

    #                 if self.num_pipeline_stages > 1:
    #                     self.send_recv_hidden(last_sending_job, (t, i))

    #                 for j in range(self.num_layers):
    #                     print('b, i, t, j = '+ str(b)+' '+str(i)+' '+str(t)+' '+str(j))
    #                     self.load_weight(b, t, i, j+1, 0)
    #                     self.load_cache(t, i, j+1, 0)
    #                     self.load_hidden(b, t, i, j, 0)
    #                     self.compute_layer(t, i, j, 0)
    #                     self.store_cache(t, i, j-1, 0)
    #                     self.store_hidden(b, t, i, j, 0)
    #                     self.sync()

    #                 last_sending_job = (t, i)

    #                 timers(timer_name).stop()

    #     if self.num_pipeline_stages > 1:
    #         self.send_recv_hidden(last_sending_job, None)
            
    # def generation_loop_overlap_single_batch(self):
    #     # Prologue
    #     for k in range(self.num_gpu_batches):
    #         self.load_weight(0, 0, k)
    #     self.sync()

    #     # Generate
    #     for i in range(self.execute_gen_len):
    #         timers("generate").start()
    #         self.update_attention_mask(i, 0)
    #         print('self.num_layers ', self.num_layers)
    #         for j in range(self.num_layers):
    #             print(' i, j = '+' '+str(i)+' '+str(j))
    #             self.load_weight(i, j+1, 0)
    #             self.load_cache(i, j+1, 0)
    #             self.load_hidden(i, j, 0)
    #             self.compute_layer(i, j, 0)
    #             self.store_cache(i, j-1, 0)
    #             self.store_hidden(i, j, 0)
    #             self.sync()
    #         timers("generate").stop()

    #         if self.task.stop and np.all(self.stopped):
    #             break

    # def generate(self,
    #              inputs: Union[np.array, List[List[int]]],
    #              max_new_tokens: int = 32,
    #              do_sample: bool = False,
    #              temperature: float = 1.0,
    #              stop: Optional[int] = None,
    #              debug_mode: Optional[str] = None,
    #              cut_gen_len: Optional[int] = None,
    #              verbose: int = 0):
    #     task = Task(
    #         inputs=inputs,
    #         prompt_len=len(inputs[0]),
    #         gen_len=max_new_tokens,
    #         cut_gen_len=cut_gen_len,
    #         do_sample=do_sample,
    #         temperature=temperature,
    #         stop=stop,
    #     )

    #     num_layers = self.num_layers
    #     num_gpu_batches = self.num_gpu_batches
    #     gpu_batch_size = self.policy.gpu_batch_size
    #     overlap = self.policy.overlap
    #     prompt_len, gen_len = task.prompt_len, task.gen_len
    #     self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len

    #     # Output token ids
    #     self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
    #         self.config.pad_token_id, dtype=np.int32)
    #     self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
    #     self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
    #     assert gpu_batch_size * num_gpu_batches == len(task.inputs)

    #     # Intermediate tensors
    #     # The following buffers store values used
    #     # for the i-th token, j-th layer, k-th gpu batch.
    #     num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
    #     for j in range(num_layers):
    #         for k in range(num_gpu_batches):
    #             self.cache_home[j][k].clear()
    #             self.cache_read_buf[j][k].clear()
    #             self.cache_write_buf[j][k].clear()
    #     for j in range(num_layers):
    #         self.weight_read_buf[j].clear()
    #     for k in range(num_gpu_batches):
    #         self.attention_mask[k].clear()
    #     self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)

    #     # Init cache
    #     self.set_task(task)
    #     for j in range(num_layers):
    #         for k in range(num_gpu_batches):
    #             self.init_cache(j, k)
    #     if self.policy.cpu_cache_compute:
    #         self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)

    #     # Generate
    #     if debug_mode is None:
    #         if not overlap:
    #             # No overlap, easy to understand, suitable for debugging
    #             self.generation_loop_normal()
    #         else:
    #             # Overlap I/O and compute
    #             if num_gpu_batches == 1:
    #                 self.generation_loop_overlap_single_batch()
    #             else:
    #                 self.generation_loop_overlap_multi_batch()
    #     elif debug_mode == "fewer_batch":
    #         # Run fewer layeres and batches for debugging
    #         if num_gpu_batches == 1:
    #             self.generation_loop_debug_single_batch()
    #         else:
    #             self.generation_loop_debug_multi_batch()
    #     elif debug_mode == "breakdown":
    #         # No overlap, fewer batches, execution time breakdown
    #         self.generation_loop_debug_normal()
    #     else:
    #         raise ValueError("Invalid debug mode: {debug_mode}")

    #     # Delete cache
    #     for j in range(num_layers):
    #         for k in range(num_gpu_batches):
    #             self.delete_cache(j, k)
    #     if self.policy.cpu_cache_compute:
    #         self.env.cpu.del_attention_compute_workspace()
# 
        # return self.output_ids