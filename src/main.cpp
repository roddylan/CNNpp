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
    xt::xarray<int> a{s};

    xt::random::seed(CONSTANT::SEED);
    // xt::random::rand()
    double bound = 1 / std::sqrt(4);
    a = xt::random::rand(
        a.shape(),
        -bound,
        bound
    );

    std::cout << a << std::endl;
}