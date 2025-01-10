cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/CNNpp)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

project(CNNpp VERSION 0.0.1)

# add libraries
# include_directories(${LIB_DIR})
# include_directories(${INC_DIR})
target_include_directories(dev PRIVATE ${LIB_DIR})
target_include_directories(dev PUBLIC ${INC_DIR})

set(HEADER_FILES ${INC_DIR}/utils.hpp ${INC_DIR}/consants.hpp ${LIB_DIR}/engine.hpp)
# set(TEMP_SOURCES ${SRC_DIR}/main.cpp )
# set(TEMP_SOURCES ${SRC_DIR}/main.cpp )
# target_include_directories(dev PRIVATE ${LIB_DIR})
# target_include_directories(dev PRIVATE ${INC_DIR})


# TODO: train + predict exec
add_executable(dev src/main.cpp ${HEADER_FILES})

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
