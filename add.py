import triton 
import triton.language as tl
import torch 

@triton.jit
def _add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        N,
        BLOCK_SIZE:tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0,BLOCK_SIZE)
    mask = offset < N
    x = tl.load(x_ptr + offset,mask=mask)
    y = tl.load(y_ptr + offset,mask=mask)

    output = x + y
    tl.store(output_ptr + offset,output,mask=mask)


def add(x,y):
    assert x.shape==y.shape
    assert x.device ==y.device

    N = x.numel()
    BLOCK_SIZE = 4096
    grid = lambda meta: (triton.cdiv(N,meta['BLOCK_SIZE']),)
    output = torch.empty_like(x)

    _add_kernel[grid](x,y,output,N,BLOCK_SIZE)
    return output

def add_python(x,y):
    return x+y

def add_torchscript(x,y):
    return torch.add(x,y)

length = [100000, 1000000, 10000000, 50000000, 100000000, 200000000, 500000000]
for i in length:
    # Test the code
    x = torch.randn(i, device="cuda",dtype=torch.float32)
    y = torch.randn(i, device="cuda",dtype=torch.float32)
    # print(f"{x=}")
    # print(f"{y=}")
    # z = add(x, y)
    # print(f"{z=}")

    # Verify correctness
    # expected = x + y
    # print(f"\nCorrect: {torch.allclose(z, expected)}")

    # Benchmarking
    import time
    start_time = time.time()
    for _ in range(10):
        z = add(x, y)
    triton_time = (time.time() - start_time) / 10
    print(f"Triton add time for length {i}: {triton_time*1000:.6f} ms")

    start_time = time.time()
    for _ in range(10):
        z = add_python(x, y)
    python_time = (time.time() - start_time) / 10
    print(f"Python add time for length {i}: {python_time*1000:.6f} ms")

    start_time = time.time()
    for _ in range(10):
        z = add_torchscript(x, y)
    torchscript_time = (time.time() - start_time) / 10
    print(f"TorchScript add time for length {i}: {torchscript_time*1000:.6f} ms")
    print("-" * 50)
