#include <benchmark/benchmark.h>
#include "tensor.hpp"

Tensor xcorr(const Tensor& input, const Tensor& kernel);
Tensor convolve(const Tensor& input, const Tensor& kernel);

static void BM_xcorr(benchmark::State& state) {
    int sz = state.range(0);
    Tensor input(sz, sz, sz);
    Tensor kernel(3, 3,3);
    for (auto _ : state) {
        benchmark::DoNotOptimize(xcorr(input, kernel));
    }
}
BENCHMARK(BM_xcorr)->Arg(4096)->Arg(8192)->Arg(16384);

static void BM_convolve(benchmark::State& state) {
    int sz = state.range(0);
    Tensor input(sz, sz, sz);
    Tensor kernel(3, 3, 3);
    for (auto _ : state) {
        benchmark::DoNotOptimize(convolve(input, kernel));
    }
}
BENCHMARK(BM_convolve)->Arg(4096)->Arg(8192)->Arg(16384);

BENCHMARK_MAIN();