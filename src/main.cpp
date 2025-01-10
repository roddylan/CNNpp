#include <cmath>
#include <iostream>
#include <vector>
#include "constants.hpp"
#include "engine.hpp"
#include "utils.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xtensor_forward.hpp"

int main() {
    cnn::Perceptron net{10};
    net.addLayer(Activation::none);

    std::vector<size_t> s{2,2};
    xt::xarray<float> a{s};

    xt::random::seed(CONSTANT::SEED);
    std::cout << a << std::endl;
    // xt::random::rand()
    double bound = 1 / std::sqrt(4);
    std::cout << "bound=" << bound << std::endl;
    a = xt::random::rand(
        a.shape(),
        -bound,
        bound
    );

    std::cout << a << std::endl;
}