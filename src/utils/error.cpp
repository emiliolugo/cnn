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

Tensor& mae(Tensor& outputs, Tensor& expected, bool requires_grad= false){
  Tensor* ret = new Tensor(outputs.dms(), requires_grad);
  size_t sz = ret->numel();

    if(requires_grad){
        ret->parents() = {outputs, expected};
        ret->backward  = mae_backwards;
    }

    for(size_t i = 0; i < sz; ++i){
        ret->set_flat(i, expected.flat_at(i) - outputs.flat_at(i));
    }
    return *ret;


}

Tensor mae_backwards(Tensor& self, BackwardCtx& ctx){
    Tensor* ret = new Tensor(self.dms());
    size_t sz = ret->numel();
    for(size_t i = 0; i < sz; ++i){
        float sum = self.flat_at(i);
        ret->set_flat(i, (sum == 0 ? 0 : sum > 0 ? 1 : -1) * 1.0f/sz);
    }
    self.parents()[0].grad = ret;

    return *ret;
}
