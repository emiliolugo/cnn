#pragma once
#include <stdexcept>
#include <vector>

struct Matrix{
    std::size_t rows;
    std::size_t cols;
    std::vector<float> arr;

    Matrix(std::size_t r, std::size_t c)
    : rows(r), cols(c), arr(r * c)
    {}

    Matrix(int r, int c)
    : rows(r), cols(c), arr(r * c)
    {}

    float& at(std::size_t r, std::size_t c) {
        return arr[r * cols + c];
    }

    const float& flat_at(std::size_t i) const {
        return arr[i];
    }

    const float& at(std::size_t r, std::size_t c) const {
        return arr[r * cols + c];
    }

    const size_t& row() const {
        return rows;
    }

    const size_t& col() const {
        return cols;
    }

    void set(std::size_t r, std::size_t c, float f){
        arr[r * cols + c] = f;
    }

    bool operator ==(const Matrix& other){
        if(rows != other.rows || cols != other.cols) return false;

        for(size_t i = 0; i < rows * cols; ++i){
            if(flat_at(i) != other.flat_at(i)) return false;
        }
        return true;
    }

    Matrix convolve(const Matrix& A, const Matrix& B);
    Matrix xcorr(const Matrix& A, const Matrix& B);


};