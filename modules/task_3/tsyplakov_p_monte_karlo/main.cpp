// Copyright 2022 Tsyplakov Pavel
#include <gtest/gtest.h>
#include <math.h>
#include <chrono>
#include <functional>
#include <iostream>
#include <vector>
#include "./monte_karlo.h"

const int amountOfPoints = 10000000;

TEST(MonteKarloSequential, Test_X_On_X) {
  std::vector<double> upperLimit = {10};
  std::vector<double> lowerLimit = {1};

  std::function<double(std::vector<double> x)> integrableFunction =
      [](std::vector<double> x) { return x[0] * x[0]; };

  auto start = std::chrono::system_clock::now();
  auto seqResult = getSequentialMonteKarlo(integrableFunction, upperLimit,
                                           lowerLimit, amountOfPoints);
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> diff = end - start;
  std::cout << std::endl
            << "Sequential time is " << diff.count() << " s." << std::endl;

  start = std::chrono::system_clock::now();
  auto parallelResult = getParallelMonteKarlo(integrableFunction, upperLimit,
                                              lowerLimit, amountOfPoints);
  end = std::chrono::system_clock::now();

  diff = end - start;
  std::cout << "Parallel time is " << diff.count() << " s." << std::endl
            << std::endl;

  ASSERT_NEAR(seqResult, parallelResult, 4);
}

TEST(MonteKarloSequential, Test_Sin_X_On_Y_On_Y) {
  std::vector<double> upperLimit = {10, 10};
  std::vector<double> lowerLimit = {1, 1};

  std::function<double(std::vector<double> x)> integrableFunction =
      [](std::vector<double> x) { return sin(x[0] * x[1] * x[1]); };

  auto start = std::chrono::system_clock::now();
  auto seqResult = getSequentialMonteKarlo(integrableFunction, upperLimit,
                                           lowerLimit, amountOfPoints);
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> diff = end - start;
  std::cout << std::endl
            << "Sequential time is " << diff.count() << " s." << std::endl;

  start = std::chrono::system_clock::now();
  auto parallelResult = getParallelMonteKarlo(integrableFunction, upperLimit,
                                              lowerLimit, amountOfPoints);
  end = std::chrono::system_clock::now();

  diff = end - start;
  std::cout << "Parallel time is " << diff.count() << " s." << std::endl
            << std::endl;

  ASSERT_NEAR(seqResult, parallelResult, 4);
}

TEST(MonteKarloSequential, Test_Cos_X_On_Cos_X_On_Y_On_3_On_Z_On_Z) {
  std::vector<double> upperLimit = {10, 10, 10};
  std::vector<double> lowerLimit = {1, 1, 1};

  std::function<double(std::vector<double> x)> integrableFunction =
      [](std::vector<double> x) {
        return cos(x[0]) * cos(x[0]) * x[1] * 3 * x[2] * x[2];
      };

  auto start = std::chrono::system_clock::now();
  auto seqResult = getSequentialMonteKarlo(integrableFunction, upperLimit,
                                           lowerLimit, amountOfPoints);
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> diff = end - start;
  std::cout << std::endl
            << "Sequential time is " << diff.count() << " s." << std::endl;

  start = std::chrono::system_clock::now();
  auto parallelResult = getParallelMonteKarlo(integrableFunction, upperLimit,
                                              lowerLimit, amountOfPoints);
  end = std::chrono::system_clock::now();

  diff = end - start;
  std::cout << "Parallel time is " << diff.count() << " s." << std::endl
            << std::endl;

  ASSERT_NEAR(seqResult, parallelResult, 20000);
}

TEST(MonteKarloSequential, Test_Cos_X_On_Sin_X_On_3_On_Y_On_Z_On_V) {
  std::vector<double> upperLimit = {10, 10, 10, 10};
  std::vector<double> lowerLimit = {1, 1, 1, 1};

  std::function<double(std::vector<double> x)> integrableFunction =
      [](std::vector<double> x) {
        return cos(x[0]) * sin(x[0]) * 3 * x[1] * x[2] * x[3];
      };

  auto start = std::chrono::system_clock::now();
  auto seqResult = getSequentialMonteKarlo(integrableFunction, upperLimit,
                                           lowerLimit, amountOfPoints);
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> diff = end - start;
  std::cout << std::endl
            << "Sequential time is " << diff.count() << " s." << std::endl;

  start = std::chrono::system_clock::now();
  auto parallelResult = getParallelMonteKarlo(integrableFunction, upperLimit,
                                              lowerLimit, amountOfPoints);
  end = std::chrono::system_clock::now();

  diff = end - start;
  std::cout << "Parallel time is " << diff.count() << " s." << std::endl
            << std::endl;

  ASSERT_NEAR(seqResult, parallelResult, 20000);
}

TEST(MonteKarloSequential, Test_10_Dim_Multiply) {
  std::vector<double> upperLimit = {10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
  std::vector<double> lowerLimit = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  std::function<double(std::vector<double> x)> integrableFunction =
      [](std::vector<double> x) {
        return x[0] * sin(x[1]) * cos(x[2]) * log(x[3]) * exp(-x[4]) * x[5] *
               x[6] * x[7] * exp(-x[8]) * exp(-x[9]) * 0.0000001;
      };

  auto start = std::chrono::system_clock::now();
  auto seqResult = getSequentialMonteKarlo(integrableFunction, upperLimit,
                                           lowerLimit, amountOfPoints);
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> diff = end - start;
  std::cout << std::endl
            << "Sequential time is " << diff.count() << " s." << std::endl;

  start = std::chrono::system_clock::now();
  auto parallelResult = getParallelMonteKarlo(integrableFunction, upperLimit,
                                              lowerLimit, amountOfPoints);
  end = std::chrono::system_clock::now();

  diff = end - start;
  std::cout << "Parallel time is " << diff.count() << " s." << std::endl
            << std::endl;

  ASSERT_NEAR(seqResult, parallelResult, 400);
}
