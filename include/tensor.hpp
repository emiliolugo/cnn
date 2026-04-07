#pragma once
#include <array>
#include <stdexcept>
#include <vector>

struct Tensor;
typedef Tensor (*backwards_func)(Tensor& self, Tensor& A, Tensor& B, float alpha);
struct Tensor{
    std::array<size_t,4> dims;

    std::vector<float> arr;
    backwards_func backward;
    
    Tensor(std::array<size_t,4> dims_)
    : dims(dims_), arr(dims[0] * dims[1] * dims[2] * dims[3])
    {}

    Tensor(size_t d0, size_t d1, size_t d2, size_t d3)
    : dims{d0, d1, d2, d3}, arr(d0 * d1 * d2 * d3)
    {}

    Tensor(std::array<int,4> dims_)
    : dims{(size_t)dims_[0], (size_t)dims_[1], (size_t)dims_[2], (size_t)dims_[3]},
      arr(dims[0] * dims[1] * dims[2] * dims[3])
    {}

    float& at(size_t i0, size_t i1, size_t i2, size_t i3) {
        return arr[i0*(dims[1]*dims[2]*dims[3])
                  + i1*(dims[2]*dims[3])
                  + i2*(dims[3])
                  + i3];
    }
    const float& flat_at(std::size_t i) const {
        return arr[i];
    }

    const float& at(size_t i0, size_t i1, size_t i2, size_t i3) const {
        return arr[i0*(dims[1]*dims[2]*dims[3])
                  + i1*(dims[2]*dims[3])
                  + i2*(dims[3])
                  + i3];
    }

    const std::array<size_t,4>& dms() const { return dims; }

    size_t batch()    const { return dims[0]; }
    size_t channels() const { return dims[1]; }
    size_t rows()     const { return dims[2]; }
    size_t cols()     const { return dims[3]; }


    void set(size_t i0, size_t i1, size_t i2, size_t i3, float f) {
        arr[i0*(dims[1]*dims[2]*dims[3])
                  + i1*(dims[2]*dims[3])
                  + i2*(dims[3])
                  + i3] = f;
    }

    bool operator ==(const Tensor& other){
        for(size_t i = 0; i < 4; ++i){
            if(dims[i]!=other.dims[i]) return false;
        }

        for(size_t i = 0; i < dims[0] * dims[1] * dims[2] * dims[3]; ++i){
            if(flat_at(i) != other.flat_at(i)) return false;
        }
        return true;
    }

    float& operator ()(size_t i0, size_t i1, size_t i2, size_t i3) {
        return arr[i0*(dims[1]*dims[2]*dims[3])
                  + i1*(dims[2]*dims[3])
                  + i2*(dims[3])
                  + i3];
    }

    Tensor convolve(const Tensor& input, const Tensor& kernel);
    Tensor xcorr(const Tensor& input, const Tensor& convolve);


};