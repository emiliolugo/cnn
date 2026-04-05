#pragma once
#include <stdexcept>
#include <vector>

struct Tensor{
    std::size_t rows_;
    std::size_t cols_;
    std::size_t depth_;
    std::vector<float> arr;

    Tensor(std::size_t r, std::size_t c, std::size_t d)
    : rows_(r), cols_(c), depth_(d), arr(r * c * d)
    {}

    Tensor(int r, int c, int d)
    : rows_(r), cols_(c), depth_(d), arr(r * c * d)
    {}

    float& at(std::size_t r, std::size_t c, std::size_t d) {
        return arr[r * cols_ + c * (rows_ * depth_) + d];
    }

    const float& flat_at(std::size_t i) const {
        return arr[i];
    }

    const float& at(std::size_t r, std::size_t c, std::size_t d) const {
        return arr[r * cols_ + c * (rows_ * depth_) + d];
    }

    const size_t& rows() const {
        return rows_;
    }

    const size_t& cols() const {
        return cols_;
    }

    const size_t& depth() const {
        return depth_;
    }

    void set(std::size_t r, std::size_t c, std::size_t d, float f){
        arr[r * cols_ + c * (rows_ * depth_) + d] = f;
    }

    bool operator ==(const Tensor& other){
        if(rows_ != other.rows_ || cols_ != other.cols_ || depth_ != other.depth_) return false;

        for(size_t i = 0; i < rows_ * cols_ * depth_; ++i){
            if(flat_at(i) != other.flat_at(i)) return false;
        }
        return true;
    }

    Tensor convolve(const Tensor& input, const Tensor& kernel);
    Tensor xcorr(const Tensor& input, const Tensor& convolve);


};