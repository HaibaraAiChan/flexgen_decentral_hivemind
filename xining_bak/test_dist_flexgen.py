"""Complete sentences with FlexGen and OPT models."""
import argparse
import pytest

from flexgen.dist_flex_opt import *
from transformers import AutoTokenizer
import os

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

    # Initialize environment
    env = ExecutionEnv.create(args.offload_dir)

    # from dist_flex_opt.py

    gpu = TorchDevice(f"cuda:{args.local_rank}")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(args.offload_dir, None, args.local_rank)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))
    TorchTensor.name_count = count(start=args.rank, step=args.world_size)

    comm_test_dist(gpu.dev if args.comm_device == "gpu" else cpu.dev, args.world_size)

    # Offloading policy
    policy = Policy(len(prompts), 1,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    overlap=True, sep_layer=True, pin_weight=args.pin_weight,
                    cpu_cache_compute=args.cpu_cache_compute, attn_sparsity=1.0,
                    compress_weight=args.compress_weight,
                    comp_weight_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=0, symmetric=False),
                    compress_cache=args.compress_cache,
                    comp_cache_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=2, symmetric=False))

    # Model
    print("Initialize...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", padding_side="left")
    tokenizer.add_bos_token = False
    stop = tokenizer("\n").input_ids[0]
    num_inner_iterations = args.num_inner_iterations if args.num_inner_iterations is not None else args.world_size
    num_inner_iterations=1
    # model = DistOptLM(args.model, env, args.path, policy)
    opt_config = get_opt_config(args.model)
    model = DistOptLM(opt_config, env, args.path, policy, args.rank,
                      args.world_size, args.comm_device, num_inner_iterations=num_inner_iterations,
                      async_comm=args.async_comm)
    args.num_gpu_batches = policy.num_gpu_batches
    args.gpu_batch_size = policy.gpu_batch_size
    num_prompts = args.num_gpu_batches * args.gpu_batch_size * num_inner_iterations * 1
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)

    cache_size = opt_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = opt_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model size: {opt_config.model_bytes()/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")


    # Generate
    print("Generate...")
    inputs = tokenizer(prompts, padding="max_length", max_length=128)
    output_ids = model.generate(
        inputs.input_ids,
        do_sample=True,
        temperature=0.7,
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
    add_distributed_parser_arguments(parser)
    args = parser.parse_args()
    # parser.add_argument("--model", type=str, default="facebook/opt-1.3b",
    #     help="The model name.")
    # parser.add_argument("--path", type=str, default="~/opt_weights",
    #     help="The path to the model weights. If there are no cached weights, "
    #          "FlexGen will automatically download them from HuggingFace.")
    # parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir",
    #     help="The directory to offload tensors. ")
    # parser.add_argument('--head-ip', type=str, default=None, help='the IP address of the head node')
    # parser.add_argument('--port', type=int, default=None, help='the port of the head node')
    # parser.add_argument('--rank', metavar='I', type=int, default=None)
    # parser.add_argument('--local-rank', metavar='I', type=int, default=None)
    # parser.add_argument('--world-size', metavar='N', type=int, default=None)
    # # parser.add_argument("--prompt-len", type=int, default=512)
    # parser.add_argument("--cut-gen-len", type=int,
    #     help="Cut generation length for fast debugging.")
    # parser.add_argument("--gen-len", type=int, default=32)
    # parser.add_argument("--percent", nargs="+", type=int,
    #     default=[100, 0, 100, 0, 100, 0],
    #     help="Six numbers. They are "
    #      "the percentage of weight on GPU, "
    #      "the percentage of weight on CPU, "
    #      "the percentage of attention cache on GPU, "
    #      "the percentage of attention cache on CPU, "
    #      "the percentage of activations on GPU, "
    #      "the percentage of activations on CPU")
    # parser.add_argument("--pin-weight", type=str2bool, nargs="?",
    #     const=True, default=True)
    # parser.add_argument('--use-mpi', action='store_true', default=True,
    #                     help="Get distributed info from MPI")
    # parser.add_argument("--cpu-cache-compute", action="store_true")
    # parser.add_argument("--compress-weight", action="store_true",
    #     help="Whether to compress weight.")
    # parser.add_argument("--compress-cache", action="store_true",
    #     help="Whether to compress cache.")
    # parser.add_argument('--comm-device', type=str, default='cpu',
    #                     choices=['gpu', 'cpu'],
    #                     help='communication through gpu nvlink or cpu memory '
    #                          'and socket')
    # # parser.add_argument("--comm-device", type=str, default="cpu")
    # parser.add_argument('--async-comm', action='store_true', default=False,
    #                     help="Use asynchronous communication")
    # parser.add_argument('--num-inner-iterations', metavar='I', type=int, default=None)
    args = parser.parse_args()
    num_gpus = torch.cuda.device_count()
    print('num_gpus ', num_gpus)
    if num_gpus>1 : 
        if args.use_mpi:
            args.world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
            args.rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
            args.local_rank = int(os.getenv('OMPI_COMM_WORLD_LOCAL_RANK'))
        initialize_distributed(args.head_ip, args.port, args.world_size,
                                args.rank, args.local_rank, args.comm_device)
    else:
        
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0
    

    assert len(args.percent) == 6

    
    try:
        run_flexgen_dist(args)
    except Exception as e:
        print(str(e))
        traceback.print_exc()
        raise e