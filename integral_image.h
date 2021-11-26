#pragma once
#include <vector>
#include <thread>
#include <atomic>
#include <fstream>
#include <opencv2/opencv.hpp>

/// Многопоточный интегратор изображений
class ImageIntegratorMT {
public:
    ImageIntegratorMT() = default;
    ImageIntegratorMT(cv::Mat& other) {
        _image = &other;
    }
    void SetImage(cv::Mat& mat) {
        _image = &mat;
    }
    void SetTarget(cv::Mat& mat) {
        _target = &mat;
    }
    void IntegrateChannel(cv::Mat& source_channel, cv::Mat& target_channel, int threads_num);
    void IntegrateImage(int threads_num);

private:
    void IntegrateRow(cv::Mat& source_channel, cv::Mat& target_channel, int row, bool prim);
    void IntegrateColumn(cv::Mat& source_channel, cv::Mat& target_channel, int row, bool prim);

    cv::Mat* _image = nullptr;          /// указатель на исходную матрицу
    cv::Mat* _target = nullptr;         /// указатель на целевую матрицу
    std::atomic_int processed_rows{0};  /// счетчик многопоточной обработки по строкам
    std::atomic_int processed_cols{0};  /// счетчик многопоточной обработки по столбцам
};

/// Функции вывода матрицы с типом данных double в файл и стандартный вывод
void ExportDoubleMatTxt(cv::Mat& image, std::string filename);
void ExportDoubleMatConsole(cv::Mat& image);
