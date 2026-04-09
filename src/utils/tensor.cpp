#include "../../include/tensor.hpp"
#include <cassert>

Tensor backward_input(const Tensor& A);
Tensor backward_kernel(const Tensor& A);

Tensor xcorr( Tensor& input, Tensor& kernels, bool requires_grad = false){
    
    auto [num_images, channels, irows, icols] = input.dms();
    auto [num_kernels, num_filters, ksz, _]   = kernels.dms();

    size_t retrows = irows -ksz -1;
    size_t retcols = icols -ksz -1;

    Tensor ret = Tensor(num_images, num_filters, retrows, retcols);

    for(size_t n = 0; n < num_images; ++n){
        for(size_t r = 0; r < retrows; ++r){
            for(size_t c = 0; c < retcols; ++c){
                for(size_t d = 0; d < num_kernels; ++d){
                    float total = 0.0;
                    for(size_t i = r; i < r + ksz; ++i){
                        for(size_t j = c; j < c + ksz; ++j){
                            for(size_t k = 0; k < channels; ++k){
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
        ret.ctx.input   = &input;
        ret.ctx.kernels = &kernels;
        ret.backward    = xcbackward;
        ret.parents()   = {&input, &kernels};
    }
    
    return ret;

}


// true indicates full, 0 is valid
Tensor convolve( Tensor& input, Tensor& kernels, bool is_full, bool requires_grad = false){
    
    auto [num_images, channels, irows, icols] = input.dms();
    auto [num_kernels, num_filters, ksz, _]   = kernels.dms();
    Tensor ret;
    if(is_full){
        size_t retrows = irows +ksz -1;
        size_t retcols = icols +ksz -1;

        ret = Tensor(num_images, num_filters, retrows, retcols);

        for(size_t n = 0; n < num_images; ++n){
            for(size_t r = 0; r < retrows; ++r){
                for(size_t c = 0; c < retcols; ++c){
                    for(size_t d = 0; d < num_kernels; ++d){
                        float total = 0.0;
                        for(size_t i = 0; i < ksz; ++i){
                            for(size_t j = 0; j < ksz; ++j){
                                int in_r = (int)r - (int)(ksz-1) + (int)i;
                                int in_c = (int)c - (int)(ksz-1) + (int)j;
                                if(in_r < 0 || in_r >= (int)irows) continue;
                                if(in_c < 0 || in_c >= (int)icols) continue;
                                for(size_t k = 0; k < channels; ++k){
                                    total += input.at(n, k, in_r, in_c)
                                           * kernels.at(d, k, ksz-1-i, ksz-1-j);
                                }
                            }
                        }
                        ret.set(n, d, r, c, total);
                    }
                }
            }
        }
    } else {
        size_t retrows = irows -ksz -1;
        size_t retcols = icols -ksz -1;

        ret = Tensor(num_images, num_filters, retrows, retcols);

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
    }


    if(requires_grad){
        ret.ctx.input   = &input;
        ret.ctx.kernels = &kernels;
        ret.backward    = xcbackward;
        ret.parents()   = {&input, &kernels};
    }
    
    return ret;

}



// ONLY FOR SINGLE CHANNEL FILTER -> NEED TO GENERALIZE!!!!
Tensor xcbackward(Tensor& self, BackwardCtx& ctx){
    Tensor& input   = *ctx.input;
    Tensor& kernels = *ctx.kernels;
    //kernel grad
    Tensor* grad = self.gd();
    Tensor kernel_grad = xcorr(input, *grad);
    //input grad
    Tensor input_grad = convolve(*grad, kernels, true); // true indicates full conv

    kernels.accumulate_grad(kernel_grad);
    return input_grad;
}

