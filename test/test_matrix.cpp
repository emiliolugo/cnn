#include "matrix.hpp"
#include <cstdio>
#include <cmath>

// Free functions declared in matrix.cpp
Matrix xcorr(const Matrix& input, const Matrix& kernel);
Matrix convolve(const Matrix& input, const Matrix& kernel);

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

static bool matrices_approx_eq(const Matrix& A, const Matrix& B) {
    if (A.rows != B.rows || A.cols != B.cols) return false;
    for (size_t i = 0; i < A.rows * A.cols; ++i) {
        if (!approx_eq(A.flat_at(i), B.flat_at(i))) return false;
    }
    return true;
}

// ── operator== tests ─────────────────────────────────────────────────────────

static void test_eq_identical() {
    Matrix A(2, 2);
    A.set(0,0,1); A.set(0,1,2);
    A.set(1,0,3); A.set(1,1,4);

    Matrix B(2, 2);
    B.set(0,0,1); B.set(0,1,2);
    B.set(1,0,3); B.set(1,1,4);

    check(A == B, "identical matrices are equal");
}

static void test_eq_different_values() {
    Matrix A(2, 2);
    A.set(0,0,1); A.set(0,1,2);
    A.set(1,0,3); A.set(1,1,4);

    Matrix B(2, 2);
    B.set(0,0,1); B.set(0,1,2);
    B.set(1,0,3); B.set(1,1,9);   // differs at (1,1)

    check(!(A == B), "matrices with different values are not equal");
}

static void test_eq_different_rows() {
    Matrix A(2, 3);
    Matrix B(3, 3);
    check(!(A == B), "matrices with different row counts are not equal");
}

static void test_eq_different_cols() {
    Matrix A(3, 2);
    Matrix B(3, 3);
    check(!(A == B), "matrices with different col counts are not equal");
}

static void test_eq_zero_matrices() {
    // default-constructed values are zero (vector<float> zero-initialises)
    Matrix A(3, 3);
    Matrix B(3, 3);
    check(A == B, "two zero-initialised matrices of same size are equal");
}

static void test_eq_single_element() {
    Matrix A(1, 1);
    Matrix B(1, 1);
    A.set(0,0,42.0f);
    B.set(0,0,42.0f);
    check(A == B, "1x1 matrices with same value are equal");

    B.set(0,0,-42.0f);
    check(!(A == B), "1x1 matrices with different values are not equal");
}

// ── xcorr tests ───────────────────────────────────────────────────────────────

// Helper: build Matrix from row-major initialiser list
static Matrix make(size_t r, size_t c, std::initializer_list<float> vals) {
    Matrix M(r, c);
    size_t idx = 0;
    for (float v : vals) M.arr[idx++] = v;
    return M;
}

// A 1x1 kernel of value 1 is an identity for xcorr.
// xcorr(input, [[1]]) == input
static void test_xcorr_identity_kernel() {
    Matrix input = make(3, 3, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });
    Matrix kernel = make(1, 1, {1.0f});
    Matrix result = xcorr(input, kernel);

    Matrix expected = make(3, 3, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });

    check(result.rows == 3 && result.cols == 3,
          "xcorr identity kernel: output size is 3x3");
    check(matrices_approx_eq(result, expected),
          "xcorr identity kernel: values match input");
}

// Manual 3x3 input, 2x2 kernel, expected 2x2 output.
//
// input:          kernel:
//  1  2  3         1  0
//  4  5  6         0  1
//  7  8  9
//
// xcorr slides kernel over input (no flip):
//   (0,0): 1*1 + 2*0 + 4*0 + 5*1 = 6
//   (0,1): 2*1 + 3*0 + 5*0 + 6*1 = 8
//   (1,0): 4*1 + 5*0 + 7*0 + 8*1 = 12
//   (1,1): 5*1 + 6*0 + 8*0 + 9*1 = 14
static void test_xcorr_2x2_kernel() {
    Matrix input = make(3, 3, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });
    Matrix kernel = make(2, 2, {
        1, 0,
        0, 1
    });
    Matrix result = xcorr(input, kernel);

    check(result.rows == 2 && result.cols == 2,
          "xcorr 2x2 kernel: output size is 2x2");

    Matrix expected = make(2, 2, {6, 8, 12, 14});
    check(matrices_approx_eq(result, expected),
          "xcorr 2x2 kernel: values match expected");
}

// All-ones kernel sums a patch of the input.
static void test_xcorr_all_ones_kernel() {
    Matrix input = make(3, 3, {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    });
    Matrix kernel = make(2, 2, {1, 1, 1, 1});
    Matrix result = xcorr(input, kernel);

    // Each output element should be 4 (sum of a 2x2 patch of 1s)
    Matrix expected = make(2, 2, {4, 4, 4, 4});

    check(result.rows == 2 && result.cols == 2,
          "xcorr all-ones: output size is 2x2");
    check(matrices_approx_eq(result, expected),
          "xcorr all-ones: each element equals 4");
}

// 3x3 kernel on a 4x4 input -> 2x2 output
static void test_xcorr_output_size_4x4_input_3x3_kernel() {
    Matrix input = make(4, 4, {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
       13,14,15,16
    });
    Matrix kernel = make(3, 3, {
        0,0,0,
        0,1,0,
        0,0,0
    });   // identity (picks centre element)
    Matrix result = xcorr(input, kernel);

    check(result.rows == 2 && result.cols == 2,
          "xcorr 3x3 kernel on 4x4 input: output is 2x2");

    // Centre elements: (1,1)=6, (1,2)=7, (2,1)=10, (2,2)=11
    Matrix expected = make(2, 2, {6, 7, 10, 11});
    check(matrices_approx_eq(result, expected),
          "xcorr centre-pick kernel: values are centre elements of input");
}

// xcorr with a zero kernel always produces zero output
static void test_xcorr_zero_kernel() {
    Matrix input = make(3, 3, {
        5, 3, 1,
        2, 8, 4,
        6, 7, 9
    });
    Matrix kernel = make(2, 2, {0, 0, 0, 0});
    Matrix result = xcorr(input, kernel);

    Matrix expected = make(2, 2, {0, 0, 0, 0});
    check(matrices_approx_eq(result, expected),
          "xcorr zero kernel: output is all zeros");
}

// ── convolve tests ───────────────────────────────────────────────────────────
//
// convolve(input, kernel) == xcorr(input, rotate180(kernel))
// rotate180: reverse all elements in row-major order.

// Test 1: symmetric kernel under 180° rotation → convolve == xcorr
//
// kernel [[1,0],[0,1]] rotated 180° is [[1,0],[0,1]] (same).
// So convolve(input, kernel) must equal xcorr(input, kernel).
//
// Using same 3x3 input / 2x2 kernel as the xcorr test above:
//   expected output: {6, 8, 12, 14}
static void test_convolve_symmetric_kernel_equals_xcorr() {
    Matrix input = make(3, 3, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });
    Matrix kernel = make(2, 2, {
        1, 0,
        0, 1
    });
    Matrix result  = convolve(input, kernel);
    Matrix expected = make(2, 2, {6, 8, 12, 14});

    check(result.rows == 2 && result.cols == 2,
          "convolve symmetric kernel: output size is 2x2");
    check(matrices_approx_eq(result, expected),
          "convolve symmetric kernel: equals xcorr (rotation leaves kernel unchanged)");
}

// Test 2: asymmetric kernel — convolve differs from xcorr.
//
// kernel [[1,2],[3,4]], rotate180 → [[4,3],[2,1]]
//
// input:           rotated kernel:
//  1  2  3          4  3
//  4  5  6          2  1
//  7  8  9
//
// (0,0): 1*4+2*3+4*2+5*1 = 4+6+8+5   = 23
// (0,1): 2*4+3*3+5*2+6*1 = 8+9+10+6  = 33
// (1,0): 4*4+5*3+7*2+8*1 = 16+15+14+8= 53
// (1,1): 5*4+6*3+8*2+9*1 = 20+18+16+9= 63
static void test_convolve_asymmetric_2x2_kernel() {
    Matrix input = make(3, 3, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });
    Matrix kernel = make(2, 2, {
        1, 2,
        3, 4
    });
    Matrix result   = convolve(input, kernel);
    Matrix expected = make(2, 2, {23, 33, 53, 63});

    check(result.rows == 2 && result.cols == 2,
          "convolve asymmetric 2x2: output size is 2x2");
    check(matrices_approx_eq(result, expected),
          "convolve asymmetric 2x2: values match xcorr with rotated kernel");

    // Sanity: result must differ from raw xcorr (kernel is not symmetric)
    Matrix xcorr_result = xcorr(input, kernel);
    check(!matrices_approx_eq(result, xcorr_result),
          "convolve asymmetric 2x2: result differs from plain xcorr");
}

// Test 3: 1x1 kernel — rotation is a no-op, output is input scaled by kernel.
static void test_convolve_1x1_kernel_scales_input() {
    Matrix input = make(3, 3, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });
    Matrix kernel = make(1, 1, {3.0f});
    Matrix result  = convolve(input, kernel);
    Matrix expected = make(3, 3, {
        3,  6,  9,
       12, 15, 18,
       21, 24, 27
    });

    check(result.rows == 3 && result.cols == 3,
          "convolve 1x1 kernel: output size is 3x3");
    check(matrices_approx_eq(result, expected),
          "convolve 1x1 kernel: output is input scaled by kernel value");
}

// Test 4: zero kernel → all-zero output regardless of input.
static void test_convolve_zero_kernel() {
    Matrix input = make(3, 3, {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1
    });
    Matrix kernel = make(2, 2, {0, 0, 0, 0});
    Matrix result  = convolve(input, kernel);
    Matrix expected = make(2, 2, {0, 0, 0, 0});

    check(matrices_approx_eq(result, expected),
          "convolve zero kernel: output is all zeros");
}

// Test 5: 3x3 asymmetric kernel on 4x4 input → 2x2 output.
//
// kernel:           rotate180:
//  1  2  3           9  8  7
//  4  5  6    →      6  5  4
//  7  8  9           3  2  1
//
// input (4x4):
//  1  2  3  4
//  5  6  7  8
//  9 10 11 12
// 13 14 15 16
//
// (0,0) patch top-left 3x3 dotted with rotated kernel:
//   1*9+2*8+3*7 + 5*6+6*5+7*4 + 9*3+10*2+11*1
//   = 9+16+21 + 30+30+28 + 27+20+11 = 192
//
// (0,1) patch cols 1-3:
//   2*9+3*8+4*7 + 6*6+7*5+8*4 + 10*3+11*2+12*1
//   = 18+24+28 + 36+35+32 + 30+22+12 = 237
//
// (1,0) patch rows 1-3:
//   5*9+6*8+7*7 + 9*6+10*5+11*4 + 13*3+14*2+15*1
//   = 45+48+49 + 54+50+44 + 39+28+15 = 372
//
// (1,1):
//   6*9+7*8+8*7 + 10*6+11*5+12*4 + 14*3+15*2+16*1
//   = 54+56+56 + 60+55+48 + 42+30+16 = 417
static void test_convolve_3x3_kernel_on_4x4_input() {
    Matrix input = make(4, 4, {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    });
    Matrix kernel = make(3, 3, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    });
    Matrix result   = convolve(input, kernel);
    Matrix expected = make(2, 2, {192, 237, 372, 417});

    check(result.rows == 2 && result.cols == 2,
          "convolve 3x3 on 4x4: output size is 2x2");
    check(matrices_approx_eq(result, expected),
          "convolve 3x3 on 4x4: values match xcorr with rotated kernel");
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    printf("=== operator== tests ===\n");
    test_eq_identical();
    test_eq_different_values();
    test_eq_different_rows();
    test_eq_different_cols();
    test_eq_zero_matrices();
    test_eq_single_element();

    printf("\n=== xcorr tests ===\n");
    test_xcorr_identity_kernel();
    test_xcorr_2x2_kernel();
    test_xcorr_all_ones_kernel();
    test_xcorr_output_size_4x4_input_3x3_kernel();
    test_xcorr_zero_kernel();

    printf("\n=== convolve tests ===\n");
    test_convolve_symmetric_kernel_equals_xcorr();
    test_convolve_asymmetric_2x2_kernel();
    test_convolve_1x1_kernel_scales_input();
    test_convolve_zero_kernel();
    test_convolve_3x3_kernel_on_4x4_input();

    printf("\n%d passed, %d failed\n", pass_count, fail_count);
    return fail_count > 0 ? 1 : 0;
}
