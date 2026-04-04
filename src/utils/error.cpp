#include <vector>
#include <cmath>
#include <cstddef>

double mae(const std::vector<double>& x, const std::vector<double>& y) {
    const std::size_t len = x.size();
    if(y.size() != len) return -1;

    double total = 0.0;
    for(std::size_t i = 0; i < len; ++i){
        total+= std::abs(y[i]-x[i]);
    }
    return total/len;
}