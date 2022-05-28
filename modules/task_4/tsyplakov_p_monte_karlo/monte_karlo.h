// Copyright 2022 Tsyplakov Pavel
#ifndef MODULES_TASK_4_TSYPLAKOV_P_MONTE_KARLO_MONTE_KARLO_H_
#define MODULES_TASK_4_TSYPLAKOV_P_MONTE_KARLO_MONTE_KARLO_H_

#include <functional>
#include <vector>

double getSequentialMonteKarlo(
    std::function<double(std::vector<double>)> const& integrableFunction,
    const std::vector<double>& upperLimit,
    const std::vector<double>& lowerLimit,
    const int amountOfPoint);

double getParallelMonteKarlo(
    std::function<double(std::vector<double>)> const& integrableFunction,
    const std::vector<double>& upperLimit,
    const std::vector<double>& lowerLimit,
    const int amountOfPoint);

#endif  // MODULES_TASK_4_TSYPLAKOV_P_MONTE_KARLO_MONTE_KARLO_H_
