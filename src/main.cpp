#include <cmath>
#include <iostream>
#include <vector>
#include "constants.hpp"
#include "engine.hpp"
#include "utils.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xtensor_forward.hpp"
#include "xtensor-blas/xlinalg.hpp"

int main() {
    cnn::Perceptron net{10};
    net.addLayer(Activation::none);

    std::vector<size_t> s{2,2};
    // xt::xarray<float> a{s, {1, 1, 1, 1}};
    // xt::xarray<float> b{s, {1, 1, 1, 1}};
    xt::xarray<int> a = {1, 1};
    // a.reshape({2, 1});
    xt::xarray<int> b = {{2, 2}};

    // xt::random::seed(CONSTANT::SEED);
    // std::cout << a << std::endl;
    // // xt::random::rand()
    // double bound = 1 / std::sqrt(4);
    // std::cout << "bound=" << bound << std::endl;
    // a = xt::random::rand(
    //     a.shape(),
    //     -bound,
    //     bound
    // );

    std::cout << "a=" << a << std::endl;
    std::cout << "b=" << b << std::endl;
    // std::cout << b * a << std::endl;
    // std::cout << xt::linspace(0, 10, 100) << std::endl;
    std::cout << "a.shape -> (" << a.shape(0) << ", " << a.shape(1) << ")\n";
    std::cout << "b.shape -> (" << b.shape(0) << ", " << b.shape(1) << ")\n\n";
    
    
    std::cout << "a.T=\n" << xt::transpose(a) << std::endl << std::endl;
    std::cout << "b.T=\n" << xt::transpose(b) << std::endl << std::endl;
}