cmake_minimum_required(VERSION 3.20)
project(prism_token_space_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(prism_token_space_native SHARED
    src/main.cpp
)

target_include_directories(prism_token_space_native PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
