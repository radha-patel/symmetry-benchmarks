#include "taco.h"
// #include "taco/format.h"
// #include "taco/lower/lower.h"
// #include "taco/ir/ir.h"
#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include "../../deps/SparseRooflineBenchmark/src/benchmark.hpp"

namespace fs = std::filesystem;

using namespace taco;

int main(int argc, char **argv)
{
    auto params = parse(argc, argv);
    Tensor<double> A = read(fs::path(params.input) / "A.ttx", Format({Dense, Sparse, Sparse}, {2, 1, 0}), true);
    Tensor<double> B_T = read(fs::path(params.input) / "B_T.ttx", Format({Dense, Dense}, {1, 0}), true);
    int n = A.getDimension(0);
    int r = B_T.getDimension(0);
    Tensor<double> C_T("C_T", {r, n}, Format({Dense, Dense}, {1, 0}));

    IndexVar i, j, k, l;

    C_T(j, i) += A(i, k, l) * B_T(j, l) * B_T(j, k);

    C_T.compile();

    // Assemble output indices and numerically compute the result
    auto time = benchmark(
        [&C_T]()
        {
            C_T.setNeedsAssemble(true);
            C_T.setNeedsCompute(true);
        },
        [&C_T]()
        {
            C_T.assemble();
            C_T.compute();
        });

    write(fs::path(params.input) / "C_T.ttx", C_T);

    json measurements;
    measurements["time"] = time;
    measurements["memory"] = 0;
    std::ofstream measurements_file(fs::path(params.output) / "measurements.json");
    measurements_file << measurements;
    measurements_file.close();
    return 0;
}
