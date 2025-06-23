import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import BaselineMLP, CustomCUDAMLP, CustomTritonMLP


def benchmark_model(model, dataloader, training=False, warmup=1, reps=5):
    device = next(model.parameters()).device
    model.train(training)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    correct = 0
    total_samples = 0

    # Warm-up
    for i, (x, y) in enumerate(dataloader):
        if i >= warmup:
            break
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)
        if training:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model(x)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(reps):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            total_samples += x.size(0)

            start.record()
            if training:
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    out = model(x)
            end.record()

            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms

    avg_time_ms = sum(times) / len(times)
    total_time_s = sum(times) / 1000.0
    throughput = total_samples / total_time_s
    accuracy = correct / total_samples * 100

    return avg_time_ms, throughput, accuracy


def print_benchmark(name, avg_ms, throughput, accuracy):
    print(
        f"{name:<24} | Latency: {avg_ms:.2f} ms | Throughput: {throughput:,.0f} samples/sec | Accuracy: {accuracy:.2f}%"
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MNIST Dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=2)

    input_dim = 28 * 28
    hidden_dim = 1024
    out_dim = 10

    baseline = BaselineMLP(input_dim, hidden_dim, out_dim).to(device)
    cuda_mlp = CustomCUDAMLP(input_dim, hidden_dim, out_dim).to(device)
    triton_mlp = CustomTritonMLP(input_dim, hidden_dim, out_dim).to(device)

    print("\n--- Inference Benchmark ---")
    t, th, acc = benchmark_model(baseline, dataloader, training=False)
    print_benchmark("Baseline MLP (PyTorch)", t, th, acc)

    t, th, acc = benchmark_model(cuda_mlp, dataloader, training=False)
    print_benchmark("Custom CUDA MLP", t, th, acc)

    t, th, acc = benchmark_model(triton_mlp, dataloader, training=False)
    print_benchmark("Custom Triton MLP", t, th, acc)

    print("\n--- Training Benchmark ---")
    t, th, acc = benchmark_model(baseline, dataloader, training=True)
    print_benchmark("Baseline MLP (PyTorch)", t, th, acc)

    t, th, acc = benchmark_model(cuda_mlp, dataloader, training=True)
    print_benchmark("Custom CUDA MLP", t, th, acc)

    t, th, acc = benchmark_model(triton_mlp, dataloader, training=True)
    print_benchmark("Custom Triton MLP", t, th, acc)
