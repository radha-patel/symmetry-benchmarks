using MatrixDepot
using BenchmarkTools
using ArgParse
using DataStructures
using JSON
using SparseArrays
using Printf
using LinearAlgebra
using Finch

#include("symv_mkl.jl")
include("ssymv_finch.jl")
include("ssymv_taco.jl")

symmetric_oski = [
    "Boeing/ct20stif",
    "Simon/olafu",
    "Boeing/bcsstk35",
    "Boeing/crystk02",
    "Boeing/crystk03",
    "Nasa/nasasrb",
    "Simon/raefsky4",
    "Mulvey/finan512",
    "Cote/vibrobox",
    "HB/saylr4",
]

unsymmetric_oski = [
    "Simon/raefsky3",
    "Simon/venkat01", 
    "FIDAP/ex11",
    "Zitney/rdist1",
    "HB/orani678",
    "Goodwin/rim",
    "Hamm/memplus",
    "HB/gemat11",
    "Mallya/lhr10",
    "Goodwin/goodwin",
    "Grund/bayer02",
    "Grund/bayer10",
    "Brethour/coater2",
    "ATandT/onetone2",
    "Wang/wang4",
    "HB/lnsp3937",
    "HB/sherman5",
    "HB/sherman3",
    "Shyy/shyy161", 
    "Wang/wang3",
]

methods = Dict(
    "ssymv_finch_ref" => ssymv_finch_ref,
#    "ssymv_mkl" => symv_mkl,
    "ssymv_finch_opt" => ssymv_finch_opt,
    "ssymv_taco" => ssymv_taco,
)

results = []
for (symmetric, dataset) in [(true, symmetric_oski), (false, unsymmetric_oski)]
    for mtx in dataset 
        A = SparseMatrixCSC(matrixdepot(mtx))
        if !symmetric
            A += transpose(A)
        end
        (n, n) = size(A)
        x = rand(n)
        y = zeros(n)
        y_ref = nothing
        for (key, method) in methods
            @info "testing" key mtx
            res = method(y, A, x)
            time = res.time
            y_res = nothing
            try
                y_res = res.y.y
            catch
                y_res = res.y
            end
            y_ref = something(y_ref, y_res)
            norm(y_res - y_ref)/norm(y_ref) < 0.1 || throw("Incorrect result via norm")

            @info "results" time
            push!(results, OrderedDict(
                "time" => time,
                "method" => key,
                "matrix" => mtx,
            ))
            write("ssymv_results.json", JSON.json(results, 4))
        end
    end
end
