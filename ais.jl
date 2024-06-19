using CUDA
using Distributions
using Random
using Plots
using BenchmarkTools

CUDA.seed!(2024)

function target_dist(x)
    return pdf(Normal(10, 1), x)
end

function proposal_dist(x)
    return pdf(Normal(0, 1), x)
end

function log_weight(x, beta)
    return beta * log(target_dist(x)) + (1 - beta) * log(proposal_dist(x))
end

function metropolis_hastings(x, x_new, beta)
    log_accept_ratio = log_weight(x_new, beta) - log_weight(x, beta)
    if log(0.5) < log_accept_ratio
        return x_new``
    else
        return x
    end
end

# GPU kernel
function ais_kernel(samples, weights, beta, N, M)
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx <= N
        @inbounds x = samples[idx]
        for i in 2:M
            x_new = x + randn() 
            @inbounds x = metropolis_hastings(x, x_new, beta[i])
            @inbounds weights[idx] += (log_weight(x, beta[i-1]) - log_weight(x, beta[i]))
        end
    end
    return
end

function ais_gpu(N, M)
    samples = CuArray(randn(N))
    weights = CuArray(CUDA.zeros(Float64, N))
    
    beta = CuArray(collect(range(0.0, stop=1.0, length=M)))
    
    
    threads_per_block = 1024
    num_blocks = cld(N, threads_per_block)
    
    @cuda threads=threads_per_block blocks=num_blocks ais_kernel(samples, weights, beta, N, M)

    weights .= exp.(weights)
    ret = weights .* samples
    return CUDA.collect(ret)
end

function ais_cpu(N, M)
    samples = randn(N)
    weights = zeros(Float64, N)
    
    betas = collect(range(0.0, stop=1.0, length=M))
    
    Threads.@threads for idx in 1:N
        x = samples[idx]
        for i in 2:M
            x_new = x + randn()  # Propose a new sample
            x = metropolis_hastings(x, x_new, betas[i])
            weights[idx] += (log_weight(x, betas[i-1]) - log_weight(x, betas[i]))
        end
    end
    
    weights .= exp.(weights)
    ret = weights .* samples
    return ret
end

function benchmark_ais(N_values, M)
    cpu_times = []
    gpu_times = []
    for N in N_values
        @show N
        cpu_time = BenchmarkTools.@belapsed ais_cpu($N, $M)
        gpu_time = BenchmarkTools.@belapsed ais_gpu($N, $M)
        push!(cpu_times, cpu_time)
        push!(gpu_times, gpu_time)
    end
    return cpu_times, gpu_times
end

function single_benchmark()
    N = 10000
    M = 10
    @btime ais_cpu(N,M)
    @btime ais_gpu(N,M)
end

function gen_plots()
    N_values = [100, 1000, 10000, 100000, 1000000, 10^7, 10^8]
    M = 10

    cpu_times, gpu_times = benchmark_ais(N_values, M)

    if !isdir("figure")
        mkdir("figure")
    end

    plt = plot(N_values, cpu_times, label="CPU", xlabel="N", ylabel="Time (s)", title="AIS Runtime Comparison: CPU vs GPU", legend=:topleft)
    plot!(N_values, gpu_times, label="GPU")
    savefig(plt, "figure/ais_scaling.png")
end