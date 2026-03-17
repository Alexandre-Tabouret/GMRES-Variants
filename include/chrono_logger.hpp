#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

class Chrono_logger {
public:
    using clock = std::chrono::steady_clock;
    using time_point = clock::time_point;
    using seconds = std::chrono::duration<double>;

    double checkpoint() {
	auto now = clock::now();
        times_.push_back(now);

	if (times_.size() > 1) {
	    return std::chrono::duration_cast<seconds>(
                now - times_[times_.size() - 2]).count();
	}
	return 0.0;
    }

    void clear() {
	times_.clear();
    }

    double get_duration(size_t start, size_t end) {
	if ((start < end) && (end < times_.size())) {
	    return std::chrono::duration_cast<seconds>(
                times_[end] - times_[start]).count();
	}
	return 0.0;
    }

private:
    std::vector<time_point> times_;
};
