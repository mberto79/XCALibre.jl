include("benchmark_2D_cylinder_case.jl")

# Edit this list to control which benchmark configurations are run in one Julia process.
# Running several entries here amortizes package compilation and method compilation costs.
benchmark_configs = [
    (backend="cpu", mode="baseline", iterations=5, warmup_iterations=1),
    (backend="cpu", mode="amg_example", iterations=5, warmup_iterations=1),
]

if CUDA.functional()
    append!(benchmark_configs, [
        (backend="cuda", mode="baseline", iterations=1, warmup_iterations=1),
        (backend="cuda", mode="amg_example", iterations=1, warmup_iterations=1),
    ])
end

run_benchmarks(benchmark_configs)
