#ifndef UTILS_HPP
#define UTILS_HPP
#include "xtensor/xtensor_forward.hpp"
#include <functional>
#include <unordered_map>
#include <cmath>
#include <iostream>
#include <xtensor.hpp>


enum class Activation {
    none, // default (linear)
    sigmoid, relu, tanh
};


// TODO: change to structs?
// Activation functors
namespace AFunctors {
    const std::function<double(float)> linear = [](float val) {
        return val;
    };

    const std::function<double(float)> sigmoid = [](float val) {
        double num, denom;
        if (val < 0) {
            num = std::exp(val);
            denom = 1 + std::exp(val);
        } else {
            num = 1;
            denom = 1 + std::exp(-val);

        }
        return num / denom;
    };

    const std::function<double(float)> relu = [](float val) {
        if (val <= 0) {
            return 0.0;
        } else {
            return static_cast<double>(val);
        }
    };
    
    // TODO: finish tanh
    const std::function<double(float)> tanh = [](float val) {
        std::cout << "\nALERT: TANH FUNCTION UNDEFINED\n";
        return 0;
    };
}

// TODO: finish backprop functions
// backprop functors
namespace BFunctors {
    const std::function<double(float)> linear = [](float val) {
        std::cout << "\nALERT: LINEAR BACKPROP/GRADIENT FUNCTION UNDEFINED\n";
        return 0;
    };

    const std::function<double(float)> sigmoid = [](float val) {
        std::cout << "\nALERT: SIGMOID BACKPROP/GRADIENT FUNCTION UNDEFINED\n";
        return 0;
    };

    const std::function<double(float)> relu = [](float val) {
        std::cout << "\nALERT: RELU BACKPROP/GRADIENT FUNCTION UNDEFINED\n";
        return 0;
    };

    const std::function<double(float)> tanh = [](float val) {
        std::cout << "\nALERT: TANH BACKPROP/GRADIENT FUNCTION UNDEFINED\n";
        return 0;
    };
}


// maps
namespace actutils {
    const std::unordered_map<Activation, std::function<double(float)>> activation{
        {Activation::none, AFunctors::linear},
        {Activation::relu, AFunctors::relu},
        {Activation::sigmoid, AFunctors::sigmoid},
        {Activation::tanh, AFunctors::tanh},
    };

    const std::unordered_map<Activation, std::function<double(float)>> backprop{
        {Activation::none, BFunctors::linear},
        {Activation::relu, BFunctors::relu},
        {Activation::sigmoid, BFunctors::sigmoid},
        {Activation::tanh, BFunctors::tanh},
    };
}

#endif