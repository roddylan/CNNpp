#include "engine.hpp"

// cnn Layer
cnn::Layer::Layer(size_t in_dim, size_t out_dim, std::function<double(float)> activation) 
        : in_dim{in_dim}, out_dim{out_dim}, activation{activation} {
    // TODO set backprop
    std::cout << "\nALERT: backprop not set\n";
}


cnn::Layer::Layer(size_t in_dim, size_t out_dim, Activation activation) 
        : in_dim{in_dim}, out_dim{out_dim} {
    this->activation = actutils::activation.at(activation);
    this->backprop = actutils::backprop.at(activation);
}


std::tuple<size_t, size_t> cnn::Layer::getDimensions() const {
    return std::tuple<size_t, size_t>{in_dim, out_dim};
}


xt::xarray<float> cnn::Layer::getWeights() const {
    return weights;
}

// cnn perceptron
cnn::Perceptron::Perceptron() : n_layers{}, layers{}, features{} {}

void cnn::Perceptron::addLayer(Activation activation) {
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