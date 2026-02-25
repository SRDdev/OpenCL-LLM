// This file contains all the required helper functions.
#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <fstream>
#include <streambuf>
#include <iostream>

// Helper to load kernel file with basic error checking
inline std::string readKernelFile(const std::string& filename) {
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open kernel file at " << filename << std::endl;
        std::cerr << "Check your relative paths!" << std::endl;
        exit(1);
    }
    
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
};

#endif