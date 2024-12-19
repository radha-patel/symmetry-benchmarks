using Finch
using TensorMarket
using JSON
function bellman_taco_helper(args, A, x)
    A = set_fill_value!(A, 0.0)
    mktempdir(prefix="input_") do tmpdir
        A_path = joinpath(tmpdir, "A.ttx")
        x_path = joinpath(tmpdir, "x.ttx")
        y_path = joinpath(tmpdir, "y.ttx")
        fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A))
        fwrite(x_path, Tensor(Dense(Element(0.0)), x))
        taco_path = joinpath(@__DIR__, "../../deps/taco/build/lib")
        withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"$taco_path", "LD_LIBRARY_PATH" => "$taco_path", "TACO_CFLAGS" => "-O3 -ffast-math -std=c99 -march=native -ggdb") do
            bellman_path = joinpath(@__DIR__, "bellman_taco")
            run(`$bellman_path -i $tmpdir -o $tmpdir $args`)
        end
        y = fread(y_path)
        time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
        return (;time=time*10^-9, y=y)
    end
end

bellman_taco(y, A, x) = bellman_taco_helper("", A, x)