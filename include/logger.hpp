
#pragma once

#include "utils.hpp"
#include <iostream>
#include <string>
#include <fstream>

using namespace utils;

struct ChronoEntry {
    int iteration;
    double math;
    double sketching; // 0
    double precond; // 1
    double spmv; // 2
    double ortho; // 3
    double facto; // 4
    double update; // 5
    double BE; //6
};

class Logger {
public:
    // Constructor
    Logger(string file_chro) {
	file_chrono.open(file_chro, std::ofstream::out | std::ofstream::trunc);
        if (!file_chrono) {
            std::cerr << "Error: Could not open or create log file: " << file_chro << std::endl;
        }
	file_chrono << "Iteration Math Sketch Precond SpMV Ortho Facto Update BE" << std::endl;
    }

    // Destructor
    ~Logger() {
	flush();
	if (file_chrono.is_open()) {
	    file_chrono.close();
	}
    }

    void log_chrono(const int iteration, const double math, const double sketching, const double precond,
	const double spmv, const double ortho, const double facto, const double update, const double be) {
    	chrono_log.push_back({iteration, math, sketching, precond, spmv, ortho, facto, update, be});
    } 
   

    void flush() {
	write_chrono();
	file_chrono.flush();
    }

private:
    std::ofstream file_chrono;
    std::vector<ChronoEntry> chrono_log;

    void write_chrono() {
        for (const auto& e : chrono_log) {
            file_chrono << e.iteration << " "
                          << e.math << " "
                          << e.sketching << " "
			  << e.precond << " "
                          << e.spmv << " "
                          << e.ortho << " "
			  << e.facto << " "
			  << e.update << " "
                          << e.BE << "\n";
        }
        chrono_log.clear();
    }

};
