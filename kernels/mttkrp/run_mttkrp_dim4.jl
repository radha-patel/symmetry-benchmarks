using MatrixDepot
using BenchmarkTools
using ArgParse
using DataStructures
using JSON
using SparseArrays
using Printf
using LinearAlgebra
using Finch

include("mttkrp_finch_dim4.jl")
include("mttkrp_taco_dim4.jl")
include("mttkrp_splatt_dim4.jl")

n = 100
rank_sparsity = [(10, 0.1), (10, 0.0001), (100, 0.1), (100, 0.0001)]
methods = Dict(
    "mttkrp_finch_ref" => mttkrp_finch_ref_dim4,
    "mttkrp_finch_opt" => mttkrp_finch_opt_dim4,
    "mttkrp_taco" => mttkrp_dim4_taco,
    "mttkrp_splatt" => mttkrp_splatt_dim4
)

results = []
N = 4
for (r, sp) in rank_sparsity
    triA = fsprand(n, n, n, n, sp)
    A_coords = unique(map(x->sort(collect(x)), zip(ffindnz(triA)[1:N]...)))
    A = fsparse((map(coord -> coord[r], symA_coords) for r = 1:N)..., rand(length(symA_coords)), tuple((n for _ in 1:N)...))
    # A = bspread("../../data/symmetric_4dim_n$(n)_sp$(sp).bsp.h5")
    B = rand(n, r)   
    C = zeros(n, r)
    C_ref = nothing
    for (key, method) in methods
        @info "testing" key n sp r
        res = method(C, A, B)
        time = res.time
        C_res = nothing
        nondiag_time = nothing
        diag_time = nothing
        try
            nondiag_time = res.nondiag_time
            diag_time = res.diag_time
        catch
            nondiag_time = nothing
            diag_time = nothing
        end
        try
            C_res = res.C.C
        catch
            C_res = res.C
        end
        C_ref = something(C_ref, C_res)
        norm(C_res - C_ref)/norm(C_ref) < 0.1 || throw("Incorrect result via norm")

        @info "results" time
        push!(results, OrderedDict(
            "time" => time,
            "nondiag_time" => nondiag_time,
            "diag_time" => diag_time,
            "method" => key,
            "sparsity" => sp,
            "size" => n,
            "rank" => r,
        ))
        write("mttkrp_dim4_results.json", JSON.json(results, 4))
    end
end