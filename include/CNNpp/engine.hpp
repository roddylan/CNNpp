#ifndef ENGINE_HPP_
#define ENGINE_HPP_

#include "xtensor/xtensor_forward.hpp"
#include <iostream>
#include <xtensor.hpp>
#include <functional>


namespace cnn {
    class Layer {
    public:
        Layer() = default;

    private:
        size_t in_dim, out_dim;
        std::function<double()> backprop;
        
        // weight matrix
        // TODO: change to tensor or tensor_fixed (if possible)
        xt::xarray<float> weights;
    };


    class CNN {
        
    };
}
#endif