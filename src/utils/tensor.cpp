#include "../../include/tensor.hpp"
#include <cassert>

Tensor backward_input(const Tensor& A);
Tensor backward_kernel(const Tensor& A);

Tensor xcorr(const Tensor& input, Tensor& kernels, bool requires_grad = false){
    
    std::array<size_t,4> input_dims = input.dms();
    size_t *num_images = &input_dims[0];
    size_t *channels   = &input_dims[1];
    size_t *irows      = &input_dims[2];
    size_t *icols      = &input_dims[3];

    std::array<size_t,4> kernels_dims = input.dms();
    size_t *num_kernels = &kernels_dims[0];
    size_t *num_filters   = &kernels_dims[1];
    size_t *ksz      = &kernels_dims[2];

    size_t retrows = irows -ksz -1;
    size_t retcols = icols -ksz -1;

    Tensor ret = Tensor(*num_images, *num_filters, retrows, retcols);

    for(size_t n = 0; n < *num_images; ++n){
        for(size_t r = 0; r < retrows; ++r){
            for(size_t c = 0; c < retcols; ++c){
                for(size_t d = 0; d < *num_kernels; ++d){
                    float total = 0.0;
                    for(size_t i = r; i < r + *ksz; ++i){
                        for(size_t j = c; j < c + *ksz; ++j){
                            for(size_t k = 0; k < *channels; ++k){
                                total += input.at(n, k, i, j) * kernels.at(d, k, i-r, j-c);
                            }
                        }
                    }
                    ret.set(n, d, r, c, total);
                }
            }
        }
    }
    if(requires_grad){
       ret.grad_fn = backward(ret,input,kernels); 
    }
    
    return ret;

}


Tensor convolve(const Tensor& input, const Tensor& kernels){
    // kernels dims: [num_kernels, channels, ksz, ksz]
    // input  dims: [batch, channels, rows, cols]
    assert(kernels.rows() == kernels.cols() &&
           kernels.rows() <= input.rows() && kernels.cols() <= input.cols());

    size_t num_images  = input.batch();
    size_t channels    = input.channels();
    size_t irows       = input.rows();
    size_t icols       = input.cols();
    size_t num_kernels = kernels.batch();
    size_t ksz         = kernels.rows();
    size_t retrows     = (irows - ksz) + 1;
    size_t retcols     = (icols - ksz) + 1;

    Tensor ret = Tensor(num_images, num_kernels, retrows, retcols);

    for(size_t n = 0; n < num_images; ++n){
        for(size_t r = 0; r < retrows; ++r){
            for(size_t c = 0; c < retcols; ++c){
                for(size_t d = 0; d < num_kernels; ++d){
                    float total = 0.0;
                    for(size_t i = r; i < r + ksz; ++i){
                        for(size_t j = c; j < c + ksz; ++j){
                            for(size_t k = 0; k < channels; ++k){
                                total += input.at(n, k, i, j)
                                       * kernels.at(d, k, ksz-1-(i-r), ksz-1-(j-c));
                            }
                        }
                    }
                    ret.set(n, d, r, c, total);
                }
            }
        }
    }
    return ret;
}



// ONLY FOR SINGLE CHANNEL FILTER -> NEED TO GENERALIZE!!!!
std::array<Tensor&,2> xcbackward(Tensor& self, Tensor& input, Tensor& kernels, Tensor& grad_output){
     std::array<size_t,4> input_dims = input.dms();
    size_t *num_images = &input_dims[0];
    size_t *channels   = &input_dims[1];
    size_t *irows      = &input_dims[2];
    size_t *icols      = &input_dims[3];

    std::array<size_t,4> kernels_dims = kernels.dms();
    size_t *num_kernels = &kernels_dims[0];
    size_t *num_filters   = &kernels_dims[1];
    size_t *ksz      = &kernels_dims[2];
    //kernel returns

    Tensor kernel_grad = xcorr(input, grad_output);
    //tensor
    Tensor input_grad = convolve(grad_output,kernels);

    return {input_grad,kernel_grad};
    
}