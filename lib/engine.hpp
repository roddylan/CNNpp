#ifndef ENGINE_HPP_
#define ENGINE_HPP_

// #include "xtensor/xtensor_forward.hpp"
#include <iostream>
#include <memory>
#include <xtensor.hpp>
#include <functional>
#include <vector>
#include <tuple>
#include "utils.hpp"


namespace cnn {
    // layer
    class Layer {
    public:
        // TODO: set xarray dims
        Layer() = default;
        ~Layer();
        
        Layer(size_t in_dim, size_t out_dim, std::function<double(float)> activation);
        
        Layer(size_t in_dim, size_t out_dim, Activation activation);

        std::tuple<size_t, size_t> getDimensions() const;

        xt::xarray<float> getWeights() const;

    private:
        size_t in_dim, out_dim;                         // input and output dimensions
        std::function<double(float)> activation;        // activation functor
        std::function<double(float)> backprop;          // backprop functor
        
        // weight matrix
        // TODO: change to tensor or tensor_fixed (if possible)
        xt::xarray<float> weights;
    };

    // perceptron
    class Perceptron {
    public:
        Perceptron();

        // void addLayer(std::function<double(float)> activation);

        void addLayer(Activation activation);

        bool train();

        size_t predict();

        std::vector<xt::xarray<float>> collectWeights();

    private:
        size_t n_layers{};
        std::vector<Layer> layers;
        
        // matrix of n features for m samples (m x n) (r x c)
        // TODO: dont use ptr
        std::vector<std::unique_ptr<xt::xarray<float>>> features;

    };
}
#endif