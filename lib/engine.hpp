#ifndef ENGINE_HPP_
#define ENGINE_HPP_

// #include "xtensor/xtensor_forward.hpp"
#include "utils.hpp"
#include <iostream>
#include <memory>
#include <xtensor.hpp>
#include <functional>
#include <vector>
#include <tuple>


namespace cnn {
    class Layer {
    public:
        // TODO: set xarray dims
        Layer() = default;
        Layer(size_t in_dim, size_t out_dim, std::function<double(float)> activation) 
        : in_dim{in_dim}, out_dim{out_dim}, activation{activation} {
            // TODO set backprop
            std::cout << "\nALERT: backprop not set\n";
        }
        
        Layer(size_t in_dim, size_t out_dim, Activation activation) 
        : in_dim{in_dim}, out_dim{out_dim} {
            this->activation = actutils::activation.at(activation);
            this->backprop = actutils::backprop.at(activation);
        }

        std::tuple<size_t, size_t> getDimensions() const {
            return std::tuple<size_t, size_t>{in_dim, out_dim};
        }

        xt::xarray<float> getWeights() const {
            return weights;
        }

    private:
        size_t in_dim, out_dim;                         // input and output dimensions
        std::function<double(float)> activation;        // activation functor
        std::function<double(float)> backprop;          // backprop functor
        
        // weight matrix
        // TODO: change to tensor or tensor_fixed (if possible)
        xt::xarray<float> weights;
    };


    class CNN {
    public:
        CNN() : n_layers{}, layers{}, features{} {};

        void addLayer(std::function<double(float)> activation) {
            Layer new_layer{};
            layers.push_back((new_layer));
        }

        void addLayer(Activation activation) {
            Layer new_layer{};
            layers.push_back((new_layer));
        }

        void train() {
            std::cout << "\nALERT: TRAINING NOT YET IMPLEMENTED\n";
            return;
        }

        size_t predict() {
            std::cout << "\nALERT: PREDICTION NOT YET IMPLEMENTED\n";
            return 0;
        }

        std::vector<xt::xarray<float>> collectWeights() {
            // xt::xarray<float> res;
            std::vector<xt::xarray<float>> res;
            for (const auto &layer : layers) {
                res.push_back(layer.getWeights());
            }

            return res;
        }

    private:
        size_t n_layers{};
        std::vector<Layer> layers;
        
        // matrix of n features for m samples (m x n) (r x c)
        // TODO: dont use ptr
        std::vector<std::unique_ptr<xt::xarray<float>>> features;

    };
}
#endif