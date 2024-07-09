# import hivemind

# # INITIAL_PEERS = [
# #     # IPv4 DNS addresses
# #     "ec2-54-177-237-94.us-west-1.compute.amazonaws.com",
# #     "ec2-52-53-152-100.us-west-1.compute.amazonaws.com",
# #     # Reserved IPs
# #     "/ip4/54.177.237.94/",
# #     "/ip4/52.53.152.100/"]

# # dht = hivemind.DHT(
# #     host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
# #     initial_peers=INITIAL_PEERS, start=True)

# dht = hivemind.DHT(
#     host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
#     start=True)

# print('\n'.join(str(addr) for addr in dht.get_visible_maddrs()))
# print("Global IP:", hivemind.utils.networking.choose_ip_address(dht.get_visible_maddrs()))


# import hivemind

# # 创建对等点A
# dht_A = hivemind.DHT(
#     host_maddrs=["/ip4/0.0.0.0/tcp/40375", "/ip4/0.0.0.0/udp/48446/quic"],
#     start=True
# )

# # 打印对等点A的可见地址
# print("对等点A的地址:")
# print('\n'.join(str(addr) for addr in dht_A.get_visible_maddrs()))
# print("公共IP地址:", hivemind.utils.networking.choose_ip_address(dht_A.get_visible_maddrs()))
# 对等点A的地址:
# /ip4/10.52.2.175/tcp/40375/p2p/12D3KooWCr99xEamG1GTqwUePZ25uzTEfbQ1mj67f9JNqf9Br3N9
# /ip4/127.0.0.1/tcp/40375/p2p/12D3KooWCr99xEamG1GTqwUePZ25uzTEfbQ1mj67f9JNqf9Br3N9
# /ip4/10.52.2.175/udp/48446/quic/p2p/12D3KooWCr99xEamG1GTqwUePZ25uzTEfbQ1mj67f9JNqf9Br3N9
# /ip4/127.0.0.1/udp/48446/quic/p2p/12D3KooWCr99xEamG1GTqwUePZ25uzTEfbQ1mj67f9JNqf9Br3N9


# import hivemind

# # 使用对等点A的地址连接对等点B
# dht_B = hivemind.DHT(
#     host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
#     initial_peers=[
#         "/ip4/129.114.108.6/tcp/40375/p2p/12D3KooWPyPY3vaCgWRbYPVuksQwxah5cLyTXuXAp3HDq1BTZanT",
#         "/ip4/129.114.108.6/udp/48446/quic/p2p/12D3KooWPyPY3vaCgWRbYPVuksQwxah5cLyTXuXAp3HDq1BTZanT",
#         # "/ip4/10.52.3.142/tcp/40375/p2p/12D3KooWPyPY3vaCgWRbYPVuksQwxah5cLyTXuXAp3HDq1BTZanT",
#         # "/ip4/10.52.3.142/udp/48446/quic/p2p/12D3KooWPyPY3vaCgWRbYPVuksQwxah5cLyTXuXAp3HDq1BTZanT",
#     ],
#     start=True
# )

# # 打印对等点B的可见地址
# print("对等点B的地址:")
# print('\n'.join(str(addr) for addr in dht_B.get_visible_maddrs()))


# import hivemind

# # INITIAL_PEERS = [
# #     # IPv4 DNS addresses
# #     "ec2-54-177-237-94.us-west-1.compute.amazonaws.com",
# #     "ec2-52-53-152-100.us-west-1.compute.amazonaws.com",
# #     # Reserved IPs
# #     "/ip4/54.177.237.94/",
# #     "/ip4/52.53.152.100/"]

# # dht = hivemind.DHT(
# #     host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
# #     initial_peers=INITIAL_PEERS, start=True)

# dht = hivemind.DHT(
#     host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
#     start=True)

# print('\n'.join(str(addr) for addr in dht.get_visible_maddrs()))
# print("Global IP:", hivemind.utils.networking.choose_ip_address(dht.get_visible_maddrs()))

# import hivemind

# # 创建初始对等点
# dht = hivemind.DHT(
#     host_maddrs=["/ip4/0.0.0.0/tcp/40375", "/ip4/0.0.0.0/udp/48446/quic"],
#     start=True
# )

# # 打印初始对等点的可见地址
# print("peer A Initial DHT addresses:")
# print('\n'.join(str(addr) for addr in dht.get_visible_maddrs()))
# print("Global IP:", hivemind.utils.networking.choose_ip_address(dht.get_visible_maddrs()))

# import hivemind

# # 使用对等点A的地址连接对等点B
# dht_B = hivemind.DHT(
#     host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
#     initial_peers=[
#         "/ip4/129.114.109.60/tcp/40375/p2p/12D3KooWCr99xEamG1GTqwUePZ25uzTEfbQ1mj67f9JNqf9Br3N9",
#         "/ip4/129.114.109.60/udp/48446/quic/p2p/12D3KooWCr99xEamG1GTqwUePZ25uzTEfbQ1mj67f9JNqf9Br3N9",
#     # ]
#     start=True
# )

# # 打印对等点B的可见地址
# print("对等点B的地址:")
# print('\n'.join(str(addr) for addr in dht_B.get_visible_maddrs()))
import time
from hivemind import DHT, Server, ModuleBackend
import torch
import requests
import socket

# Set up the DHT with the other node as an initial peer
NODE_A_ADDRESS = "129.114.108.6:40375"
NODE_B_ADDRESS = "129.114.109.60:40375"

# Get private IP
private_ip = socket.gethostbyname(socket.gethostname())
print(f"Private IP: {private_ip}")

# Get public IP
try:
    public_ip = requests.get('https://api.ipify.org').text
    print(f"Public IP: {public_ip}")
except:
    print("Couldn't get public IP. Using private IP.")
    public_ip = private_ip

# Determine which node this is and set up accordingly
if public_ip == "129.114.108.6":
    print("This is Node A")
    dht = DHT(start=True, initial_peers=[NODE_B_ADDRESS])
elif public_ip == "129.114.109.60":
    print("This is Node B")
    dht = DHT(start=True, initial_peers=[NODE_A_ADDRESS])
else:
    raise ValueError(f"Unknown IP: {public_ip}")

# Dummy module for the backend
class DummyModule(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return f"Response from {public_ip} (private: {private_ip}) for input of shape {x.shape}"

# Define input schema
class TensorSchema:
    def __init__(self, shape):
        self.shape = shape

    def make_zeros(self, batch_size):
        return torch.zeros(batch_size, *self.shape)

# Initialize the module backend with a name, module, and args_schema
module_backends = [
    ModuleBackend(
        name="dummy_module",
        module=DummyModule(),
        args_schema=(TensorSchema((3, 4)),)
    )
]

# Initialize the Hivemind server with the DHT and module backends
server = Server(
    dht=dht,
    module_backends=module_backends,
    num_connection_handlers=10
)

# Run the server in the background
server.run_in_background(await_ready=True)
print(f"Hivemind server is running on {public_ip} (private: {private_ip})...")

# Keep the server running and periodically check for peers
while True:
    print(f"Current peers: {dht.get_visible_maddrs()}")
    time.sleep(60)