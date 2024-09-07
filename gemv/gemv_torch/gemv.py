import torch

# Function to generate random tensors
def gen(N, K):
    return (
        torch.rand((K, N), dtype=torch.float16).cuda(),
        torch.rand((1, K), dtype=torch.float16).cuda(),
    )

# Profiling function using torch.cuda.Event and torch.cuda.synchronize()
def profile_gemv():
    test_num = 64
    M_list = [1] * test_num
    N_list = [(i + 1) * 256 for i in range(test_num)]
    K_list = [(i + 1) * 256 for i in range(test_num)]

    # Warm-up phase
    print("Warming up...")
    warmup_B, warmup_A = gen(1024, 128)
    for _ in range(5):
        torch.matmul(warmup_A, warmup_B)
    torch.cuda.synchronize()  # Ensure all warm-up operations are done
    print("Warm-up done.\n")

    for j in range(10):
        M = M_list[j]
        N = N_list[j]
        K = 128

        # Generate the random tensors A and B
        B, A = gen(N, K)

        # Create CUDA events to record time
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Synchronize before starting the timer
        torch.cuda.synchronize()

        # Record the start event
        start_event.record()

        # Perform matrix multiplication
        C = torch.matmul(A, B)

        # Record the end event
        end_event.record()

        # Synchronize again to make sure all operations are finished
        torch.cuda.synchronize()

        # Calculate elapsed time in milliseconds
        elapsed_time_ms = start_event.elapsed_time(end_event)  # Returns time in milliseconds

        sec = elapsed_time_ms / 1000

        avg_Gflops = M * N * K * 2 / 1000 / 1000 / 1000 / sec;

        print(f"Test {j+1}: N={N}, K={K}, Elapsed time: {elapsed_time_ms:.6f} ms, Throughput: {avg_Gflops:.2f} Gflops")


if __name__ == "__main__":
    profile_gemv()