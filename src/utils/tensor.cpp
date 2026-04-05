#include "../../include/tensor.hpp"
#include <cassert>


Tensor xcorr(const Tensor& input, const Tensor& kernel){
    assert(kernel.rows() == kernel.cols() && kernel.rows() == kernel.depth() &&
           kernel.rows() <= input.rows() && kernel.rows() <= input.cols() && kernel.rows() <= input.depth());
    size_t irows= input.rows(); 
    size_t icols = input.cols();
    size_t idepth = input.depth();
    size_t ksz = kernel.rows();
    size_t retrows = (irows - ksz) + 1;
    size_t retcols = (icols - ksz) + 1;
    size_t retdepth = (idepth - ksz) + 1;
    Tensor ret = Tensor(retrows, retcols, retdepth);

    for(size_t r = 0; r < retrows; ++r){
        for(size_t c = 0; c < retcols; ++c){
            for(size_t d = 0; d < retdepth; ++d){
                float total = 0.0;
                for(size_t i = r; i < r + ksz; ++i){
                    for(size_t j = c; j < c + ksz; ++j){
                        for(size_t k = d; k < d + ksz; ++k){
                           total+= input.at(i,j,k) * kernel.at(i-r,j-c,k-d); 
                        }
                    }
                }
                ret.set(r,c,d,total);
            }
        }
    }
    return ret;
}
Tensor convolve(const Tensor& input, const Tensor& kernel){
    assert(kernel.rows() == kernel.cols() && kernel.rows() == kernel.depth() &&
           kernel.rows() <= input.rows() && kernel.rows() <= input.cols() && kernel.rows() <= input.depth());
    size_t irows= input.rows(); 
    size_t icols = input.cols();
    size_t idepth = input.depth();
    size_t ksz = kernel.rows();
    size_t retrows = (irows - ksz) + 1;
    size_t retcols = (icols - ksz) + 1;
    size_t retdepth = (idepth - ksz) + 1;
    Tensor ret = Tensor(retrows, retcols, retdepth);

    for(size_t r = 0; r < retrows; ++r){
        for(size_t c = 0; c < retcols; ++c){
            for(size_t d = 0; d < retdepth; ++d){
                float total = 0.0;
                for(size_t i = r; i < r + ksz; ++i){
                    for(size_t j = c; j < c + ksz; ++j){
                        for(size_t k = d; k < d + ksz; ++k){
                           total+= input.at(i,j,k) * kernel.at(ksz - (i-r) -1,ksz - (j-c) - 1, ksz - (k-d) - 1);
                        }
                    }
                }
                ret.set(r,c,d,total);
            }
        }
    }
    return ret;
}

