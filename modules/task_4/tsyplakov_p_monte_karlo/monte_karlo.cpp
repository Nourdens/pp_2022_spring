// Copyright 2022 Tsyplakov Pavel
#include "../../../modules/task_4/tsyplakov_p_monte_karlo/monte_karlo.h"
#include <exception>
#include <functional>
#include <random>
#include <utility>
#include <vector>
#include "../../../3rdparty/unapproved/unapproved.h"

using std::exception;
using std::function;
using std::pair;
using std::vector;

double getSequentialMonteKarlo(
    function<double(vector<double>)> const& integrableFunction,
    const vector<double>& upperLimit,
    const vector<double>& lowerLimit,
    const int amountOfPoint) {
    if (upperLimit.size() == 0 || lowerLimit.size() == 0 ||
        upperLimit.size() != lowerLimit.size()) {
        throw "Wrong limit!";
    }

    double result = 0.0;
    auto integrableDimensions = upperLimit.size();

    vector<std::uniform_real_distribution<double>> distributions(
        integrableDimensions);
    for (unsigned int counter = 0; counter < integrableDimensions; ++counter) {
        distributions[counter] = std::uniform_real_distribution<double>(
            lowerLimit[counter], upperLimit[counter]);
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    vector<double> randomPoints(integrableDimensions);
    for (int counter = 0; counter < amountOfPoint; ++counter) {
        for (unsigned int i = 0; i < integrableDimensions; ++i) {
            randomPoints[i] = distributions[i](gen);
        }

        result += integrableFunction(randomPoints);
    }

    double partialResult = 1.0;
    for (unsigned int counter = 0; counter < integrableDimensions; ++counter) {
        partialResult *= upperLimit[counter] - lowerLimit[counter];
    }

    result *= (partialResult / amountOfPoint);
    return result;
}

double getParallelMonteKarlo(
    std::function<double(std::vector<double>)> const& integrableFunction,
    const std::vector<double>& upperLimit,
    const std::vector<double>& lowerLimit,
    const int amountOfPoint) {
    if (upperLimit.size() == 0 || lowerLimit.size() == 0 ||
        upperLimit.size() != lowerLimit.size()) {
        throw "Wrong limit!";
    }

    double reductionResult = 0.0;
    auto integrableDimensions = upperLimit.size();

    vector<std::uniform_real_distribution<double>> distributions(
        integrableDimensions);
    for (unsigned int counter = 0; counter < integrableDimensions; ++counter) {
        distributions[counter] = std::uniform_real_distribution<double>(
            lowerLimit[counter], upperLimit[counter]);
    }

    int numberThread = std::thread::hardware_concurrency();
    vector<double> threadsResult(numberThread);
    std::thread* threads = new std::thread[numberThread];
    int dataPortion = amountOfPoint / numberThread;

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < numberThread; ++i) {
        pair<vector<double>::size_type, vector<double>::size_type> localPoints;
        if (i != numberThread - 1) {
            localPoints = { i * dataPortion, (i + 1) * dataPortion };
        }
        else {
            localPoints = { i * dataPortion, amountOfPoint };
        }

        threads[i] = std::thread(
            [&](const pair<vector<double>::size_type, vector<double>::size_type>&
                local,
                const int currentThread) {
                    vector<double> randomPoints(integrableDimensions);
                    for (auto counter = local.first; counter < local.second; ++counter) {
                        for (unsigned int i = 0; i < integrableDimensions; ++i) {
                            randomPoints[i] = distributions[i](gen);
                        }
                        threadsResult[currentThread] += integrableFunction(randomPoints);
                    }
            },
            localPoints, i);
    }

    for (int i = 0; i < numberThread; ++i) {
        threads[i].join();
        reductionResult += threadsResult[i];
    }
    delete[] threads;

    double partialResult = 1.0;
    for (unsigned int counter = 0; counter < integrableDimensions; ++counter) {
        partialResult *= upperLimit[counter] - lowerLimit[counter];
    }

    reductionResult *= (partialResult / amountOfPoint);
    return reductionResult;
}
