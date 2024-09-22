"""Complete sentences with FlexGen and OPT models."""
import argparse
# import pytest
import hivemind

from flexgen.dist_flex_opt import *
from flexgen.flex_decentralized import *
from flexgen.opt_config import get_opt_config
from transformers import AutoTokenizer

import torch
print(torch.cuda.is_available())
UID_DELIMITER = "."
# INITIAL_PEERS = [
#     # IPv4 DNS addresses
#     "/ec2-54-177-237-94.us-west-1.compute.amazonaws.com",
#     "/ec2-52-53-152-100.us-west-1.compute.amazonaws.com",
#     # Reserved IPs
#     "/ip4/54.177.237.94/",
#     "/ip4/52.53.152.100/"]

# INITIAL_PEERS = ["/ip4/172.31.11.38/tcp/38461/p2p/12D3KooWN3VnDVjnn255X62XPTGYmF6rLUq7x9uHnbGPhH3y91iU", 
#                 "/ip4/172.31.11.38/udp/55744/quic/p2p/12D3KooWN3VnDVjnn255X62XPTGYmF6rLUq7x9uHnbGPhH3y91iU"]


dht = hivemind.DHT(host_maddrs=["/ip4/0.0.0.0/tcp/38461", "/ip4/0.0.0.0/udp/55744/quic"],
                   start=True) # node 1
print('\n'.join(str(addr) for addr in dht.get_visible_maddrs()))
print(type(dht), dht.is_alive())
print('main function dht', dht)
# dht = hivemind.DHT(host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
#                    initial_peers=INITIAL_PEERS, 
#                    start=True)   # node 2

def main(args):
    # Prompts
    prompts = [
        "Question: Where were the 2004 Olympics held?\n"
        "Answer: Athens, Greece\n"
        "Question: What is the longest river on the earth?\n"
        "Answer:",

        "Extract the airport codes from this text.\n"
        "Text: \"I want a flight from New York to San Francisco.\"\n"
        "Airport codes: JFK, SFO.\n"
        "Text: \"I want you to book a flight from Phoenix to Las Vegas.\"\n"
        "Airport codes:",
    ]
    
    # Model
    print("Initialize...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    # print('tokenizer', tokenizer)
    tokenizer.add_bos_token = False # there will not be added the "beginning of sequence token"
    # print('after add_bos_token : tokenizer', tokenizer) # 
    stop = tokenizer("\n").input_ids[0] # tokenizes the newline character, 
    # retrieves its corresponding token ID from the tokenizer's vocabulary, 
    # and assigns that ID to the variable stop. 
    # This ID can then be used in further processing, 
    # such as in model inputs or for controlling the flow of text generation.
    # print("Stop is %d\n "%stop)
    
    num_inner_iterations = args.num_inner_iterations if args.num_inner_iterations is not None else args.world_size
    # num_inner_iterations = args.num_inner_iterations
    print('args ', args)
    print('args.num_gpu_batches ', args.num_gpu_batches)
    print('args.gpu_batch_size ', args.gpu_batch_size)
    print('num_inner_iterations ', num_inner_iterations)
    num_prompts = args.num_gpu_batches * args.gpu_batch_size * num_inner_iterations * 1
    print('num_prompts ', num_prompts)
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    # Task and policy
    # warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)
    # print('inputs ', inputs)
    print('len of the inputs ', len(list(inputs)))
    print('len of the first inputs ', len(list(inputs[0])))
    gpu = TorchDevice(f"cuda:{args.local_rank}")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir, None, args.local_rank)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))
    TorchTensor.name_count = count(start=args.rank, step=args.world_size)
    # comm_test_dist(gpu.dev if args.comm_device == "gpu" else cpu.dev, args.world_size)
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"

    opt_config = get_opt_config(args.model)
    print('model name ', args.model)
    
    # Initialize environment
    # env = ExecutionEnv.create(args.offload_dir)

    # Offloading policy
    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=0, symmetric=False),
                    args.compress_cache,
                    CompressionConfig(num_bits=4, group_size=64,
                                      group_dim=2, symmetric=False))

   
    gpu = TorchDevice(f"cuda:{args.local_rank}")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir, None, args.local_rank)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))
    TorchTensor.name_count = count(start=args.rank, step=2)
    # model = DecOptLM(get_opt_config(args.model), env, args.path, policy, args.rank,
    #                   2, args.comm_device, num_inner_iterations=num_inner_iterations, dht=INITIAL_PEERS) # node 2
    # model = DecLM(get_opt_config(args.model), env, args.path, policy, device_rank=args.rank,
    #                   num_blocks=2, comm_device=args.comm_device, num_inner_iterations=num_inner_iterations, dht=dht) # node 1
    block = DecBlock(get_opt_config(args.model), env, args.path, policy, device_rank=args.rank,
                      block_idx=args.block_idx,  dht=dht) # node 1

    cache_size = opt_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = opt_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model size: {opt_config.model_bytes()/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")
    
    print("Start Generating...")

    output_ids = model.generate(
        inputs,
        do_sample=True,
        temperature=0.7,
        cut_gen_len=cut_gen_len,
        max_new_tokens=32,
        stop=stop)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print("Outputs:\n" + 70 * '-')
    for i in [0, len(outputs)-1]:
        print(f"{i}: {outputs[i]}")
        print("-" * 70)

    # Shutdown
    print("Shutdown...")
    env.close_copy_threads()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()
    
    args.world_size = 1
    args.rank = 0
    args.local_rank = 0
    args.head_ip='127.0.0.1'
    args.port = 7777
    args.model='facebook/opt-125m'
    args.cut_gen_len=5 
    args.gpu_batch_size=12
    args.comm_device = 'cpu' 
    args.sep_layer=False
    args.num_inner_iterations = 1
    # args.block_idx=[0,1,2,3,4,5,6,7,8,9,10,11] #total 12 transformer layers, one input layer, one output layer
    args.block_idx=[0,1,2,3,4,5,] #total 12 transformer layers, one input layer, one output layer
    # args.block_idx=[6,7,8,9,10,11] #total 12 transformer layers, one input layer, one output layer

    assert len(args.percent) == 6
    
    try:
        main(args)
    except Exception as e:
        print(str(e))
        # traceback.print_exc()
        raise e
    print('model.name ', args.model)
    