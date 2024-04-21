using Finch
using TensorMarket
using JSON

function spmv_mkl_helper(args, A, x)
    mktempdir(prefix="input_") do tmpdir
        A_path = joinpath(tmpdir, "A.ttx")
        x_path = joinpath(tmpdir, "x.ttx")
        y_path = joinpath(tmpdir, "y.ttx")
        fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A))
        fwrite(x_path, Tensor(Dense(Element(0.0)), x))
        run(`mkl_spmv -i $tmpdir -o $tmpdir`)
        # run(`$spmv_path -i $tmpdir -o $tmpdir $args`)
        # taco_path = joinpath(@__DIR__, "../deps/taco/build/lib")
        # withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"$taco_path", "LD_LIBRARY_PATH" => "$taco_path", "TACO_CFLAGS" => "-O3 -ffast-math -std=c99 -march=native -ggdb") do
        #     spmv_path = joinpath(@__DIR__, "spmv_taco")
        #     run(`$spmv_path -i $tmpdir -o $tmpdir $args`)
        # end
        # y = fread(y_path)
        # time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
        # return (;time=time*10^-9, y=y)
    end
end

# spmv_taco(y, A, x) = spmv_taco_helper("", A, x)

n = 100
# A = Tensor(Dense(SparseList(Element(0))), fsprand(Int, n, n, 0.1))
# x = Tensor(Dense(Element(0)), rand(Int, n))
A = rand(n, n)
x = rand(n)
spmv_mkl_helper("", A, x)
