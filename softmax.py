import torch 
import torch.nn.functional as F
import triton.language as tl
import triton
import numpy as np

@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    x = tl.load(input_ptr + offset, mask=mask)
    max_x = tl.max(x, axis=0)
    x_exp = tl.exp(x - max_x)
    sum_x_exp = tl.sum(x_exp, axis=0)
    softmax_output = x_exp / sum_x_exp

    tl.store(output_ptr + offset, softmax_output, mask=mask)


def softmax(input_tensor, BLOCK_SIZE=1024):
    assert input_tensor.device.type == 'cuda', "Input tensor must be on CUDA device"

    N = input_tensor.numel()
    # BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    output_tensor = torch.empty_like(input_tensor)

    _softmax_kernel[grid](input_tensor, output_tensor, N, BLOCK_SIZE)
    return output_tensor

def softmax_python(input_tensor):
    exp_x = np.exp(input_tensor.cpu().numpy() - np.max(input_tensor.cpu().numpy()))
    softmax_output = exp_x / np.sum(exp_x)
    return torch.from_numpy(softmax_output).to(input_tensor.device)

# Example usage and testing

length = [100000, 1000000, 10000000, 50000000, 100000000, 200000000, 500000000]
for i in length:
    # Test the code
    x = torch.randn(i, device="cuda",dtype=torch.float32)
    import time
    start_time = time.time()
    for _ in range(10):
        z = softmax(x)
    triton_time = (time.time() - start_time) / 10
    print(f"Triton add time for length {i}: {triton_time*1000:.6f} ms")

    start_time = time.time()
    for _ in range(10):
        z = softmax_python(x)
    python_time = (time.time() - start_time) / 10
    print(f"Python add time for length {i}: {python_time*1000:.6f} ms")

    start_time = time.time()
    for _ in range(10):
        z = F.softmax(x)
    torchscript_time = (time.time() - start_time) / 10
    print(f"TorchScript add time for length {i}: {torchscript_time*1000:.6f} ms")
    print("-" * 50)


block_size_list = [256, 512, 1024, 2048, 4096, 8192]
for i in block_size_list:
    x = torch.randn(10000000, device="cuda",dtype=torch.float32)
    import time
    start_time = time.time()
    for _ in range(10):
        z = softmax(x, BLOCK_SIZE=i)
    triton_time = (time.time() - start_time) / 10
    print(f"Triton add time for block size {i}: {triton_time*1000:.6f} ms")

