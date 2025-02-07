// Copyright 2022 Ustiuzhanin Nikita

#include "../../../pp_2022_spring/modules/task_2/ustiuzhanin_n_radix_sort_simple_merge/radix_sort.h"
#include <string>

void randomVector(vector<int>* data, size_t size, size_t rad) {
    if (!size)
        return;

    data->clear();
    data->resize(size);

    std::random_device rd;
    std::mt19937 mersenne(rd());
    std::uniform_int_distribution<int> dist(0, pow(10, rad));
    for (size_t i = 0; i < data->size(); i++)
        data->at(i) = dist(mersenne);
}

int getNum(int val, size_t pos) {
    while (pos > 1) {
        val /= 10;
        pos--;
    }

    return val % 10;
}

size_t rad(size_t value) {
    size_t counter = 0;
    while (value != 0) {
        value /= 10;
        counter++;
    }
    return counter;
}

void radixSort(vector<int>* data) {
    if (!data->size())
        return;

    if (std::find_if(data->begin(), data->end(), [](int val) {
        return val < 0; }) != data->end())
        throw std::string("Try sort numbers less then 0");

    size_t maxRad = rad(*std::max_element(data->begin(), data->end()));

    vector<list<int>> helpList(10);

    for (size_t i = 1; i <= maxRad; i++) {
        for (int val : *data) {
            helpList[getNum(val, i)].push_back(val);
        }

        data->clear();
        for (list<int> l : helpList) {
            for (int val : l)
                data->push_back(val);
        }

        helpList.clear();
        helpList.resize(10);
    }
}

vector<int> simpleMerge(const vector<int>& firstVector,
    const vector<int>& secondVector) {
    vector<int> resultVector(firstVector.size() + secondVector.size());

    size_t firstIt, secondIt, resutIt;
    firstIt = secondIt = resutIt = 0;

    while (firstIt < firstVector.size() && secondIt < secondVector.size()) {
        if (firstVector[firstIt] < secondVector[secondIt]) {
            resultVector[resutIt++] = firstVector[firstIt++];
        } else {
            resultVector[resutIt++] = secondVector[secondIt++];
        }
    }

    while (firstIt < firstVector.size()) {
        resultVector[resutIt++] = firstVector[firstIt++];
    }
    while (secondIt < secondVector.size()) {
        resultVector[resutIt++] = secondVector[secondIt++];
    }

    return resultVector;
}

void radixSortOMP(vector<int>* data) {
    if (!data->size())
        return;

    if (std::find_if(data->begin(), data->end(), [](int val) {
        return val < 0; }) != data->end())
        throw std::string("Try sort numbers less then 0");

    int numberOfThread = omp_get_num_procs();
    int dataPortion = data->size() / numberOfThread;

    vector<vector<int>> vecOfVec(numberOfThread);

#pragma omp parallel num_threads(numberOfThread)
    {
        int currentThread = omp_get_thread_num();
        vector<int> local;

        if (currentThread != numberOfThread - 1) {
            local = { data->begin() + currentThread * dataPortion,
                     data->begin() + (currentThread + 1) * dataPortion };
        } else {
            local = { data->begin() + currentThread * dataPortion,
                     data->end() };
        }

        radixSort(&local);
        vecOfVec[currentThread] = local;
    }

    vector<int> resultVector = vecOfVec[0];
    for (int i = 1; i < numberOfThread; ++i) {
        resultVector = simpleMerge(resultVector, vecOfVec[i]);
    }

    *data = resultVector;
}
