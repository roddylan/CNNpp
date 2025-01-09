cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(CNNpp VERSION 0.0.1)

# TODO: train + predict exec
add_executable(dev src/main.cpp)


# add dependencies (CPM)
include(cmake/CPM.cmake)

CPMAddPackage(
    NAME xtensor
    GITHUB_REPOSITORY xtensor-stack/xtensor
    GIT_TAG 0.25.0
)

CPMAddPackage(
    NAME magic_enum
    GITHUB_REPOSITORY Neargye/magic_enum
    GIT_TAG v0.9.7
)

target_link_libraries(dev PUBLIC xtensor)
# target_link_libraries(CNN PUBLIC magic_enum)
