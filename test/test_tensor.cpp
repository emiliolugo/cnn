#include "tensor.hpp"
#include <cstdio>
#include <cmath>

Tensor xcorr(const Tensor& input, const Tensor& kernel);
Tensor convolve(const Tensor& input, const Tensor& kernel);

static int pass_count = 0;
static int fail_count = 0;

static void check(bool condition, const char* name) {
    if (condition) {
        printf("  PASS: %s\n", name);
        ++pass_count;
    } else {
        printf("  FAIL: %s\n", name);
        ++fail_count;
    }
}

static bool approx_eq(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

// ── operator== tests ──────────────────────────────────────────────────────────

static void test_eq_identical() {
    Tensor A(2, 3, 4);
    Tensor B(2, 3, 4);
    A.set(1, 2, 3, 7.0f);
    B.set(1, 2, 3, 7.0f);
    check(A == B, "eq: identical tensors are equal");
}

static void test_eq_different_values() {
    Tensor A(2, 2, 2);
    Tensor B(2, 2, 2);
    A.set(0, 0, 0, 1.0f);
    B.set(0, 0, 0, 9.0f);
    check(!(A == B), "eq: different values are not equal");
}

static void test_eq_different_rows() {
    Tensor A(2, 3, 4);
    Tensor B(3, 3, 4);
    check(!(A == B), "eq: different rows are not equal");
}

static void test_eq_different_cols() {
    Tensor A(3, 2, 4);
    Tensor B(3, 4, 4);
    check(!(A == B), "eq: different cols are not equal");
}

static void test_eq_different_depth() {
    Tensor A(3, 3, 2);
    Tensor B(3, 3, 4);
    check(!(A == B), "eq: different depth are not equal");
}

static void test_eq_zero_initialized() {
    Tensor A(3, 3, 3);
    Tensor B(3, 3, 3);
    check(A == B, "eq: zero-initialized tensors of same shape are equal");
}

static void test_eq_all_elements_set() {
    Tensor A(2, 2, 2);
    Tensor B(2, 2, 2);
    float val = 0.0f;
    for (size_t r = 0; r < 2; ++r)
        for (size_t c = 0; c < 2; ++c)
            for (size_t d = 0; d < 2; ++d) {
                A.set(r, c, d, val);
                B.set(r, c, d, val);
                val += 1.0f;
            }
    check(A == B, "eq: fully populated tensors are equal");
}

static void test_eq_one_element_differs() {
    Tensor A(2, 2, 2);
    Tensor B(2, 2, 2);
    A.set(1, 1, 1, 42.0f);
    check(!(A == B), "eq: tensors with one differing element are not equal");
}

// ── set tests ─────────────────────────────────────────────────────────────────

static void test_set_basic() {
    Tensor A(3, 3, 3);
    A.set(1, 1, 1, 5.0f);
    check(approx_eq(A.at(1, 1, 1), 5.0f), "set: value is stored correctly");
}

static void test_set_overwrites() {
    Tensor A(3, 3, 3);
    A.set(0, 0, 0, 3.0f);
    A.set(0, 0, 0, 99.0f);
    check(approx_eq(A.at(0, 0, 0), 99.0f), "set: second set overwrites first");
}

static void test_set_does_not_affect_neighbors() {
    Tensor A(3, 3, 3);
    A.set(1, 1, 1, 42.0f);
    check(approx_eq(A.at(0, 0, 0), 0.0f), "set: neighbor {0,0,0} unaffected");
    check(approx_eq(A.at(1, 1, 0), 0.0f), "set: neighbor {1,1,0} unaffected");
    check(approx_eq(A.at(1, 0, 1), 0.0f), "set: neighbor {1,0,1} unaffected");
    check(approx_eq(A.at(2, 2, 2), 0.0f), "set: neighbor {2,2,2} unaffected");
}

static void test_set_negative_value() {
    Tensor A(2, 2, 2);
    A.set(0, 1, 0, -3.14f);
    check(approx_eq(A.at(0, 1, 0), -3.14f), "set: negative float stored correctly");
}

static void test_set_all_corners() {
    Tensor A(2, 2, 2);
    A.set(0,0,0, 1); A.set(0,0,1, 2);
    A.set(0,1,0, 3); A.set(0,1,1, 4);
    A.set(1,0,0, 5); A.set(1,0,1, 6);
    A.set(1,1,0, 7); A.set(1,1,1, 8);
    check(approx_eq(A.at(0,0,0), 1) && approx_eq(A.at(0,0,1), 2) &&
          approx_eq(A.at(0,1,0), 3) && approx_eq(A.at(0,1,1), 4) &&
          approx_eq(A.at(1,0,0), 5) && approx_eq(A.at(1,0,1), 6) &&
          approx_eq(A.at(1,1,0), 7) && approx_eq(A.at(1,1,1), 8),
          "set: all 8 corners of a 2x2x2 tensor set correctly");
}

// ── at tests ──────────────────────────────────────────────────────────────────

static void test_at_default_zero() {
    Tensor A(4, 4, 4);
    bool all_zero = true;
    for (size_t r = 0; r < 4; ++r)
        for (size_t c = 0; c < 4; ++c)
            for (size_t d = 0; d < 4; ++d)
                all_zero = all_zero && approx_eq(A.at(r, c, d), 0.0f);
    check(all_zero, "at: default-constructed tensor is all zeros");
}

static void test_at_returns_correct_value() {
    Tensor A(3, 3, 3);
    A.set(2, 1, 0, 13.0f);
    check(approx_eq(A.at(2, 1, 0), 13.0f), "at: returns value previously set");
}

static void test_at_const() {
    Tensor A(2, 2, 2);
    A.set(0, 0, 0, 5.0f);
    const Tensor& cA = A;
    check(approx_eq(cA.at(0, 0, 0), 5.0f), "at: const at returns correct value");
    check(approx_eq(cA.at(1, 1, 1), 0.0f), "at: const at returns zero for unset element");
}

static void test_at_distinct_depth_slices() {
    Tensor A(2, 2, 3);
    A.set(0, 0, 0, 10.0f);
    A.set(0, 0, 1, 20.0f);
    A.set(0, 0, 2, 30.0f);
    check(approx_eq(A.at(0,0,0), 10.0f) &&
          approx_eq(A.at(0,0,1), 20.0f) &&
          approx_eq(A.at(0,0,2), 30.0f),
          "at: depth slices at same (r,c) are independent");
}

static void test_at_flat_order() {
    Tensor A(2, 2, 2);
    float val = 1.0f;
    for (size_t r = 0; r < 2; ++r)
        for (size_t c = 0; c < 2; ++c)
            for (size_t d = 0; d < 2; ++d)
                A.set(r, c, d, val++);
    bool ok = true;
    val = 1.0f;
    for (size_t r = 0; r < 2; ++r)
        for (size_t c = 0; c < 2; ++c)
            for (size_t d = 0; d < 2; ++d)
                ok = ok && approx_eq(A.at(r, c, d), val++);
    check(ok, "at: sequential set and read back all elements");
}

// ── xcorr tests ───────────────────────────────────────────────────────────────
//
// xcorr slides the kernel over input with no flip.

// Helper: fill a tensor sequentially 1, 2, 3, ... in (r,c,d) order.
static Tensor make_seq(size_t r, size_t c, size_t d) {
    Tensor T(r, c, d);
    float v = 1.0f;
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            for (size_t k = 0; k < d; ++k)
                T.set(i, j, k, v++);
    return T;
}

static bool tensors_approx_eq(const Tensor& A, const Tensor& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols() || A.depth() != B.depth()) return false;
    for (size_t i = 0; i < A.rows() * A.cols() * A.depth(); ++i)
        if (std::fabs(A.flat_at(i) - B.flat_at(i)) > 1e-5f) return false;
    return true;
}

// 1x1x1 kernel of value 1 is an identity: xcorr(input, {1}) == input
static void test_xcorr_identity_kernel() {
    Tensor input = make_seq(3, 3, 3);
    Tensor kernel(1, 1, 1);
    kernel.set(0, 0, 0, 1.0f);
    Tensor result = xcorr(input, kernel);
    check(result.rows() == 3 && result.cols() == 3 && result.depth() == 3,
          "xcorr identity kernel: output size is 3x3x3");
    check(tensors_approx_eq(result, input),
          "xcorr identity kernel: output equals input");
}

// Output size: input(4,4,4) with kernel(2,2,2) -> output(3,3,3)
static void test_xcorr_output_size() {
    Tensor input(4, 4, 4);
    Tensor kernel(2, 2, 2);
    Tensor result = xcorr(input, kernel);
    check(result.rows() == 3 && result.cols() == 3 && result.depth() == 3,
          "xcorr output size: (4,4,4) input, (2,2,2) kernel -> (3,3,3)");
}

// All-zero kernel always produces zero output.
static void test_xcorr_zero_kernel() {
    Tensor input = make_seq(3, 3, 3);
    Tensor kernel(2, 2, 2);  // all zeros by default
    Tensor result = xcorr(input, kernel);
    Tensor expected(2, 2, 2);
    check(tensors_approx_eq(result, expected),
          "xcorr zero kernel: output is all zeros");
}

// All-ones 2x2x2 kernel on all-ones 3x3x3 input.
// Each output element = sum of 8 ones = 8.
static void test_xcorr_all_ones_kernel_3d() {
    Tensor input(3, 3, 3);
    for (size_t i = 0; i < 3*3*3; ++i) input.flat_at(i); // can't set via flat_at (const)
    // fill input with ones
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 3; ++c)
            for (size_t d = 0; d < 3; ++d)
                input.set(r, c, d, 1.0f);
    Tensor kernel(2, 2, 2);
    for (size_t r = 0; r < 2; ++r)
        for (size_t c = 0; c < 2; ++c)
            for (size_t d = 0; d < 2; ++d)
                kernel.set(r, c, d, 1.0f);
    Tensor result = xcorr(input, kernel);
    check(result.rows() == 2 && result.cols() == 2 && result.depth() == 2,
          "xcorr all-ones 3D: output size is 2x2x2");
    bool all_eight = true;
    for (size_t r = 0; r < 2; ++r)
        for (size_t c = 0; c < 2; ++c)
            for (size_t d = 0; d < 2; ++d)
                all_eight = all_eight && approx_eq(result.at(r,c,d), 8.0f);
    check(all_eight, "xcorr all-ones 3D: every output element equals 8");
}

// Corner-pick kernel: 2x2x2 kernel with 1 only at (0,0,0), 0 elsewhere.
// xcorr with no flip picks input(r,c,d) for each output position.
//
// input = make_seq(3,3,3):
//   (0,0,0)=1  (0,0,1)=2  (0,1,0)=4  (0,1,1)=5
//   (1,0,0)=10 (1,0,1)=11 (1,1,0)=13 (1,1,1)=14
static void test_xcorr_corner_pick_kernel() {
    Tensor input = make_seq(3, 3, 3);
    Tensor kernel(2, 2, 2);
    kernel.set(0, 0, 0, 1.0f);  // all others zero
    Tensor result = xcorr(input, kernel);
    check(result.rows() == 2 && result.cols() == 2 && result.depth() == 2,
          "xcorr corner-pick: output size is 2x2x2");
    check(approx_eq(result.at(0,0,0),  1.0f) &&
          approx_eq(result.at(0,0,1),  2.0f) &&
          approx_eq(result.at(0,1,0),  4.0f) &&
          approx_eq(result.at(0,1,1),  5.0f) &&
          approx_eq(result.at(1,0,0), 10.0f) &&
          approx_eq(result.at(1,0,1), 11.0f) &&
          approx_eq(result.at(1,1,0), 13.0f) &&
          approx_eq(result.at(1,1,1), 14.0f),
          "xcorr corner-pick: output equals top-left-front sub-region of input");
}

// Fully 3D known-value test.
// input(r,c,d) = r*9 + c*3 + d + 1  (values 1..27, 3x3x3)
// kernel(r,c,d) = r*4 + c*2 + d + 1 (values 1..8,  2x2x2)
//
// xcorr(0,0,0) = 1*1+2*2+4*3+5*4+10*5+11*6+13*7+14*8 = 356
// xcorr(0,0,1) = 2*1+3*2+5*3+6*4+11*5+12*6+14*7+15*8 = 392
static void test_xcorr_3d_known_values() {
    Tensor input = make_seq(3, 3, 3);
    Tensor kernel = make_seq(2, 2, 2);
    Tensor result = xcorr(input, kernel);
    check(result.rows() == 2 && result.cols() == 2 && result.depth() == 2,
          "xcorr 3D known: output size is 2x2x2");
    check(approx_eq(result.at(0,0,0), 356.0f),
          "xcorr 3D known: (0,0,0) == 356");
    check(approx_eq(result.at(0,0,1), 392.0f),
          "xcorr 3D known: (0,0,1) == 392");
}

// ── convolve tests ────────────────────────────────────────────────────────────
//
// convolve(input, kernel) == xcorr(input, rotate180(kernel))
// rotate180 flips all three axes: kernel_r(i,j,k) = kernel(ksz-1-i, ksz-1-j, ksz-1-k)

// 1x1x1 kernel: rotation is a no-op, so convolve == xcorr.
static void test_convolve_1x1x1_scales_input() {
    Tensor input = make_seq(3, 3, 3);
    Tensor kernel(1, 1, 1);
    kernel.set(0, 0, 0, 3.0f);
    Tensor result = convolve(input, kernel);
    check(result.rows() == 3 && result.cols() == 3 && result.depth() == 3,
          "convolve 1x1x1: output size is 3x3x3");
    // Each element should be input * 3
    Tensor expected = make_seq(3, 3, 3);
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 3; ++c)
            for (size_t d = 0; d < 3; ++d)
                expected.set(r, c, d, expected.at(r,c,d) * 3.0f);
    check(tensors_approx_eq(result, expected),
          "convolve 1x1x1: output is input scaled by kernel value");
}

// All-ones kernel is symmetric under 180° rotation, so convolve == xcorr.
static void test_convolve_symmetric_kernel_equals_xcorr() {
    Tensor input(3, 3, 3);
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 3; ++c)
            for (size_t d = 0; d < 3; ++d)
                input.set(r, c, d, 1.0f);
    Tensor kernel(2, 2, 2);
    for (size_t r = 0; r < 2; ++r)
        for (size_t c = 0; c < 2; ++c)
            for (size_t d = 0; d < 2; ++d)
                kernel.set(r, c, d, 1.0f);
    Tensor r1 = xcorr(input, kernel);
    Tensor r2 = convolve(input, kernel);
    check(tensors_approx_eq(r1, r2),
          "convolve symmetric kernel: equals xcorr");
}

// Corner-pick kernel for convolve: 2x2x2 kernel with 1 only at (1,1,1).
// convolve flips the kernel, so (1,1,1) maps to (0,0,0) after rotation.
// This means convolve picks input(r+1, c+1, d+1) for each output position.
//
// input = make_seq(3,3,3):
//   (1,1,1)=14 (1,1,2)=15 (1,2,1)=17 (1,2,2)=18
//   (2,1,1)=23 (2,1,2)=24 (2,2,1)=26 (2,2,2)=27
static void test_convolve_corner_pick_kernel() {
    Tensor input = make_seq(3, 3, 3);
    Tensor kernel(2, 2, 2);
    kernel.set(1, 1, 1, 1.0f);  // all others zero

    Tensor result = convolve(input, kernel);

    check(result.rows() == 2 && result.cols() == 2 && result.depth() == 2,
          "convolve corner-pick: output size is 2x2x2");

    // After flipping, kernel[1,1,1] -> [0,0,0]
    // So we pick the TOP-LEFT-FRONT corner of each 2x2x2 block
    check(approx_eq(result.at(0,0,0), 1.0f) &&
          approx_eq(result.at(0,0,1), 2.0f) &&
          approx_eq(result.at(0,1,0), 4.0f) &&
          approx_eq(result.at(0,1,1), 5.0f) &&
          approx_eq(result.at(1,0,0), 10.0f) &&
          approx_eq(result.at(1,0,1), 11.0f) &&
          approx_eq(result.at(1,1,0), 13.0f) &&
          approx_eq(result.at(1,1,1), 14.0f),
          "convolve corner-pick: output equals top-left-front sub-region of input");

    // Now xcorr should differ (since it does NOT flip)
    Tensor xcorr_result = xcorr(input, kernel);

    check(!tensors_approx_eq(result, xcorr_result),
          "convolve corner-pick: result differs from plain xcorr");
}

// Fully 3D known-value test.
// Same input and kernel as test_xcorr_3d_known_values above.
//
// convolve(0,0,0): uses rotated kernel -> 1*8+2*7+4*6+5*5+10*4+11*3+13*2+14*1 = 184
// convolve(0,0,1): i,j in {0,1}, k in {1,2}:
//   input(0,0,1)*k(1,1,1)+input(0,0,2)*k(1,1,0)+input(0,1,1)*k(1,0,1)+input(0,1,2)*k(1,0,0)
//   +input(1,0,1)*k(0,1,1)+input(1,0,2)*k(0,1,0)+input(1,1,1)*k(0,0,1)+input(1,1,2)*k(0,0,0)
//   = 2*8+3*7+5*6+6*5+11*4+12*3+14*2+15*1
//   = 16+21+30+30+44+36+28+15 = 220
static void test_convolve_3d_known_values() {
    Tensor input = make_seq(3, 3, 3);
    Tensor kernel = make_seq(2, 2, 2);
    Tensor result = convolve(input, kernel);
    check(result.rows() == 2 && result.cols() == 2 && result.depth() == 2,
          "convolve 3D known: output size is 2x2x2");
    check(approx_eq(result.at(0,0,0), 184.0f),
          "convolve 3D known: (0,0,0) == 184");
    check(approx_eq(result.at(0,0,1), 220.0f),
          "convolve 3D known: (0,0,1) == 220");
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    printf("=== equality tests ===\n");
    test_eq_identical();
    test_eq_different_values();
    test_eq_different_rows();
    test_eq_different_cols();
    test_eq_different_depth();
    test_eq_zero_initialized();
    test_eq_all_elements_set();
    test_eq_one_element_differs();

    printf("\n=== set tests ===\n");
    test_set_basic();
    test_set_overwrites();
    test_set_does_not_affect_neighbors();
    test_set_negative_value();
    test_set_all_corners();

    printf("\n=== at tests ===\n");
    test_at_default_zero();
    test_at_returns_correct_value();
    test_at_const();
    test_at_distinct_depth_slices();
    test_at_flat_order();

    printf("\n=== xcorr tests ===\n");
    test_xcorr_identity_kernel();
    test_xcorr_output_size();
    test_xcorr_zero_kernel();
    test_xcorr_all_ones_kernel_3d();
    test_xcorr_corner_pick_kernel();
    test_xcorr_3d_known_values();

    printf("\n=== convolve tests ===\n");
    test_convolve_1x1x1_scales_input();
    test_convolve_symmetric_kernel_equals_xcorr();
    test_convolve_corner_pick_kernel();
    test_convolve_3d_known_values();

    printf("\n%d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
