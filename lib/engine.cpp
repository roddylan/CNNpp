#include "engine.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xrandom.hpp"
#include <constants.hpp>
#include <stdexcept>
#include <tuple>

// TODO: xarray initialize
// cnn Layer
// cnn::Layer::Layer(size_t in_dim, size_t out_dim, std::function<double(float)> activation) 
//         : in_dim{in_dim}, out_dim{out_dim}, activation{activation} {
//     // TODO set backprop
//     std::cout << "\nALERT: backprop not set\n";
// }

cnn::Layer::Layer(const size_t &in_dim, const size_t &out_dim, const Activation &activation) 
        : in_dim{in_dim}, out_dim{out_dim} {
    this->activation = actutils::activation.at(activation);
    this->backprop = actutils::backprop.at(activation);
    
    // create weights matrix
    std::vector<size_t> shape = {in_dim, out_dim};
    xt::xarray<float> weights(shape);
    
    // init with random
    xt::random::seed(CONSTANT::SEED);
    // weight initializing via Xavier
    double bound = 1 / std::sqrt(in_dim);
    
    weights = xt::random::rand(weights.shape(), -bound, bound);

}


std::tuple<size_t, size_t> cnn::Layer::getDimensions() const {
    return std::tuple<size_t, size_t>{in_dim, out_dim};
}


xt::xarray<float> cnn::Layer::getWeights() const {
    return weights;
}


// cnn perceptron
cnn::Perceptron::Perceptron(size_t n_ft) : n_layers{}, layers{}, features{}, n_ft{n_ft} {}

void cnn::Perceptron::addLayer(const Activation &activation, const size_t &out) {
    // arg validation
    if (out < 1) {
        throw std::invalid_argument("Received invalid output dimension value. Must be greater than 0");
    }
    size_t in_dim{}, out_dim{out};
    
    // no layers -> in_dim = n_ft
    if (n_layers == 0) {
        in_dim = n_ft;
    } else {
        in_dim = std::get<1>(layers.back().getDimensions());
    }

    Layer new_layer{};
    layers.push_back((new_layer));
    ++n_layers;
}

// TODO: finish training and prediction code
bool cnn::Perceptron::train() {
    std::cout << "\nALERT: TRAINING NOT YET IMPLEMENTED\n";
    return false;
}

size_t cnn::Perceptron::predict() {
    std::cout << "\nALERT: PREDICTION NOT YET IMPLEMENTED\n";
    return 0;
}

std::vector<xt::xarray<float>> cnn::Perceptron::collectWeights() {
    // xt::xarray<float> res;
    std::vector<xt::xarray<float>> res;
    for (const auto &layer : layers) {
        res.push_back(layer.getWeights());
    }

    return res;
}