#include "../../include/matrix.hpp"
#include <cassert>


Matrix xcorr(const Matrix& input, const Matrix& kernel){
    assert(kernel.row() == kernel.col() && kernel.row() <= input.row());
    size_t irows= input.row(); 
    size_t icols = input.col();
    
    size_t ksz = kernel.row();
    size_t retrows = (irows - ksz) + 1;
    size_t retcols = (icols - ksz) + 1;
    Matrix ret = Matrix(retrows, retcols);

    for(size_t r = 0; r < retrows; ++r){
        for(size_t c = 0; c < retcols; ++c){
            float total = 0.0;
            for(size_t i = r; i < r + ksz; ++i){
                for(size_t j = c; j < c + ksz; ++j){
                    total+= input.at(i,j) * kernel.at(i-r,j-c);
                }
            }
            ret.set(r,c,total);
        }
    }
    return ret;
}
Matrix convolve(const Matrix& input, const Matrix& kernel){
    assert(kernel.row() == kernel.col() && kernel.row() <= input.row());
    size_t irows= input.row(); 
    size_t icols = input.col();
    
    size_t ksz = kernel.row();
    size_t retrows = (irows - ksz) + 1;
    size_t retcols = (icols - ksz) + 1;
    Matrix ret = Matrix(retrows, retcols);

    for(size_t r = 0; r < retrows; ++r){
        for(size_t c = 0; c < retcols; ++c){
            float total = 0.0;
            for(size_t i = r; i < r + ksz; ++i){
                for(size_t j = c; j < c + ksz; ++j){
                    total+= input.at(i,j) * kernel.at(ksz - (i-r) -1,ksz - (j-c) - 1);
                }
            }
            ret.set(r,c,total);
        }
    }
    return ret;
}

