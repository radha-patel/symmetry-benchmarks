#include <iostream>
#include "splatt.h"
#include "../../deps/SparseRooflineBenchmark/src/benchmark.hpp"
#include "../../deps/taco/include/taco.h"
#include "../../deps/taco/include/taco/util/timers.h"

namespace fs = std::filesystem;
using namespace taco;

#define BENCH(CODE, NAME, REPEAT, TIMER, COLD)         \
    {                                                  \
        TACO_TIME_REPEAT(CODE, REPEAT, TIMER, COLD);   \
        std::cout << NAME << " time (ms)" << std::endl \
                  << TIMER << std::endl;               \
    }

int main(int argc, char **argv){
    auto params = parse(argc, argv);

    /* allocate default options */
    double *cpd_opts = splatt_default_opts();
    cpd_opts[SPLATT_OPTION_NTHREADS] = 1;
    cpd_opts[SPLATT_OPTION_NITER] = 0;
    cpd_opts[SPLATT_OPTION_CSF_ALLOC] = SPLATT_CSF_ALLMODE;
    cpd_opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;

    /* load the tensor from a file */
    int ret;
    splatt_idx_t nmodes;
    splatt_csf *tt;
    ret = splatt_csf_load("../../data/symmetric_10x10x10.tns", &nmodes, &tt, cpd_opts);
    std::cout << "after loading A" << std::endl;

    Tensor<double> _B = read(fs::path(params.input) / "B.ttx", Format({Dense, Dense}), true);
    int n = _B.getDimension(0);
    double **factors = new double *[nmodes];
    for (int i = 0; i < nmodes; ++i) {
        factors[i] = (double *)(_B.getStorage().getValues().getData());
    }
    std::cout << "after loading B" << std::endl;

    Tensor<double> C_splatt({n, n}, Format({Dense, Dense}));
    C_splatt.pack();
    double *matout = (double *)(C_splatt.getStorage().getValues().getData());

        /* perform mttkrp */
        const int mode = 0;
    taco::util::TimeResults timevalue;
    splatt_mttkrp(mode, n, tt, factors, matout, cpd_opts);
    std::cout << "after performing mttkrp" << std::endl;

    write(fs::path(params.input) / "C.ttx", C_splatt);
    std::cout << "after writing C" << std::endl;
    /* display results */
    // std::cout << "Result vector C:" << std::endl;
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         std::cout << C_splatt(i, j) << " ";
    //     }
    // }
    // std::cout << std::endl;

        // BENCH(splatt_mttkrp(mode, n, tt, factors, (double *)(C_splatt.getStorage().getValues().getData()), cpd_opts);,
        //         "\nSPLATT", 1, timevalue, true);
        // printResults("mode-1 mttkrp", results);
    std::cout << "before returning" << std::endl;
    return 0;
    std::cout << "after returning??" << std::endl;
}