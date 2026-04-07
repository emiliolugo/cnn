#include <benchmark/benchmark.h>
#include "tensor.hpp"

Tensor xcorr(const Tensor& input, const Tensor& kernel);
Tensor convolve(const Tensor& input, const Tensor& kernel);

// state.range(0) = input side length (cubic: NxNxN)
// state.range(1) = kernel side length (cubic: KxKxK)
static void BM_xcorr(benchmark::State& state) {
    int sz = state.range(0);
    int ksz = state.range(1);
    Tensor input(sz, sz, sz);
    Tensor kernel(ksz, ksz, ksz);
    for (auto _ : state) {
        benchmark::DoNotOptimize(xcorr(input, kernel));
    }
    state.SetItemsProcessed(state.iterations() *
        (sz - ksz + 1) * (sz - ksz + 1) * (sz - ksz + 1));
}
BENCHMARK(BM_xcorr)->Args({32, 1})->Args({32, 3})->Args({32, 5})
                   ->Args({64, 1})->Args({64, 3})->Args({64, 5})
                   ->Args({128, 1})->Args({128, 3})->Args({128, 5})
                   ->Args({256, 3})->Args({256, 5});

static void BM_convolve(benchmark::State& state) {
    int sz = state.range(0);
    int ksz = state.range(1);
    Tensor input(sz, sz, sz);
    Tensor kernel(ksz, ksz, ksz);
    for (auto _ : state) {
        benchmark::DoNotOptimize(convolve(input, kernel));
    }
    state.SetItemsProcessed(state.iterations() *
        (sz - ksz + 1) * (sz - ksz + 1) * (sz - ksz + 1));
}
BENCHMARK(BM_convolve)->Args({32, 1})->Args({32, 3})->Args({32, 5})
                      ->Args({64, 1})->Args({64, 3})->Args({64, 5})
                      ->Args({128, 1})->Args({128, 3})->Args({128, 5})
                      ->Args({256, 3})->Args({256, 5});

BENCHMARK_MAIN();
