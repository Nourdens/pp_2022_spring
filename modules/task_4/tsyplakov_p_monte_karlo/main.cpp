// Copyright 2022 Tsyplakov Pavel
#include <gtest/gtest.h>
#include <math.h>
#include <functional>
#include <vector>
#include "./monte_karlo.h"

const int amountOfPoints = 10000;

TEST(MonteKarloSequential, Test_X_On_Y) {
    std::vector<double> upperLimit = { 2, 5 };
    std::vector<double> lowerLimit = { 1, 3 };

    std::function<double(std::vector<double> x)> integrableFunction =
        [](std::vector<double> x) { return x[0] * x[1]; };

    auto seqResult = getSequentialMonteKarlo(integrableFunction, upperLimit,
        lowerLimit, amountOfPoints);
    auto parallelResult = getParallelMonteKarlo(integrableFunction, upperLimit,
        lowerLimit, amountOfPoints);

    ASSERT_NEAR(seqResult, parallelResult, 2);
}

TEST(MonteKarloSequential, Test_Sin_X) {
    std::vector<double> upperLimit = { 10 };
    std::vector<double> lowerLimit = { 1 };

    std::function<double(std::vector<double> x)> integrableFunction =
        [](std::vector<double> x) { return sin(x[0]); };

    auto seqResult = getSequentialMonteKarlo(integrableFunction, upperLimit,
        lowerLimit, amountOfPoints);
    auto parallelResult = getParallelMonteKarlo(integrableFunction, upperLimit,
        lowerLimit, amountOfPoints);

    ASSERT_NEAR(seqResult, parallelResult, 2);
}

TEST(MonteKarloSequential, Test_Cos_X_On_X) {
    std::vector<double> upperLimit = { 10 };
    std::vector<double> lowerLimit = { 1 };

    std::function<double(std::vector<double> x)> integrableFunction =
        [](std::vector<double> x) { return cos(x[0]) * x[0]; };

    auto seqResult = getSequentialMonteKarlo(integrableFunction, upperLimit,
        lowerLimit, amountOfPoints);
    auto parallelResult = getParallelMonteKarlo(integrableFunction, upperLimit,
        lowerLimit, amountOfPoints);

    ASSERT_NEAR(seqResult, parallelResult, 2);
}

TEST(MonteKarloSequential, Test_Cos_X_On_Sin_X_On_X) {
    std::vector<double> upperLimit = { 10 };
    std::vector<double> lowerLimit = { 1 };

    std::function<double(std::vector<double> x)> integrableFunction =
        [](std::vector<double> x) { return cos(x[0]) * sin(x[0]) * x[0]; };

    auto seqResult = getSequentialMonteKarlo(integrableFunction, upperLimit,
        lowerLimit, amountOfPoints);
    auto parallelResult = getParallelMonteKarlo(integrableFunction, upperLimit,
        lowerLimit, amountOfPoints);

    ASSERT_NEAR(seqResult, parallelResult, 2);
}

TEST(MonteKarloSequential, Test_Cos_X_On_Sin_X_On_Sin_X_On_X) {
    std::vector<double> upperLimit = { 10 };
    std::vector<double> lowerLimit = { 1 };

    std::function<double(std::vector<double> x)> integrableFunction =
        [](std::vector<double> x) {
        return cos(x[0]) * sin(x[0]) * sin(x[0]) * x[0];
    };

    auto seqResult = getSequentialMonteKarlo(integrableFunction, upperLimit,
        lowerLimit, amountOfPoints);
    auto parallelResult = getParallelMonteKarlo(integrableFunction, upperLimit,
        lowerLimit, amountOfPoints);

    ASSERT_NEAR(seqResult, parallelResult, 2);
}
