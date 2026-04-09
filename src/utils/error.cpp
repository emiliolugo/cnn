#include <vector>
#include <cmath>
#include <cstddef>
#include "../../include/tensor.hpp"

double mae(const std::vector<double>& x, const std::vector<double>& y) {
    const std::size_t len = x.size();
    if(y.size() != len) return -1;

    double total = 0.0;
    for(std::size_t i = 0; i < len; ++i){
        total+= std::abs(y[i]-x[i]);
    }
    return total/len;
}

Tensor mae(Tensor& outputs, Tensor& expected, bool requires_grad = false){
    Tensor ret(outputs.dms(), requires_grad);
    size_t sz = ret.numel();

    for(size_t i = 0; i < sz; ++i){
        ret.set_flat(i, std::abs(expected.flat_at(i) - outputs.flat_at(i)));
    }

    if(requires_grad){
        ret.parents() = {&outputs, &expected};
        ret.backward  = mae_backwards;
    }

    return ret;
}

Tensor mae_backwards(Tensor& self, BackwardCtx& ctx){
    Tensor ret(self.dms());
    size_t sz = ret.numel();
    for(size_t i = 0; i < sz; ++i){
        float residual = self.flat_at(i); 
        float out = self.parents()[0]->flat_at(i);
        float exp = self.parents()[1]->flat_at(i);
        float sign = (out == exp) ? 0.0f : (out > exp ? 1.0f : -1.0f);
        ret.set_flat(i, sign / sz);
    }
    self.parents()[0]->accumulate_grad(ret);

    return ret;
}
