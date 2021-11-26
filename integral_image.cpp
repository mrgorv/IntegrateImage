#include <iostream>
#include "integral_image.h"
using namespace std;

/**
 * @brief IntegrateRowST интегрирование одной строки матрицы
 * @param source_channel исходный канал
 * @param target_channel целевой канал
 * @param row номер обрабатывыемой строки
 * @param prim первый ли это проход (нужно ли прибавлять значения из исходного канала)
 */
void ImageIntegratorMT::IntegrateRow(cv::Mat& source_channel, cv::Mat& target_channel, int row, bool prim) {
	for (int j = 0; j < source_channel.cols; ++j) {
		if (prim) target_channel.at<double>(row, j) = source_channel.at<double>(row, j);
		if (j != 0) {
			target_channel.at<double>(row, j) += target_channel.at<double>(row, j - 1);
		}
	}
    processed_rows++;
}

/**
 * @brief IntegrateColumnST интегрирование одного столбца матрицы (аналогично предыдущему)
 */
void ImageIntegratorMT::IntegrateColumn(cv::Mat& source_channel, cv::Mat& target_channel, int col, bool prim) {
	for (int i = 0; i < source_channel.rows; ++i) {
		if (prim) target_channel.at<double>(i, col) = source_channel.at<double>(i, col);
		if (i != 0) {
			target_channel.at<double>(i, col) += target_channel.at<double>(i - 1, col);
		}
	}
    processed_cols++;
}

/**
 * @brief IntegrateChannelMT Многопоточное получение интегрального изображение для одного канала.
 * Интегрирование сначала ведется по строкам блоками по количеству доступных потоков, затем аналогично по столбцам.
 * Потоки объединяются после выполнения каждого блока.
 * Интегральным для пустого канала автоматически считается аналогичный пустой канал.
 * @param source_channel исходный канал
 * @param target_channel целевой канал
 * @param threads_num количество потоков
 */
void ImageIntegratorMT::IntegrateChannel(cv::Mat& source_channel, cv::Mat& target_channel, int threads_num) {
    if (source_channel.empty()) {
        target_channel = source_channel;
        return;
    }
    processed_rows = 0;
    processed_cols = 0;
    while (processed_rows < source_channel.rows) {
		std::vector<thread> row_threads;
        for (int i = processed_rows; i < min(threads_num + processed_rows, source_channel.rows); ++i) {
            row_threads.push_back(thread(&ImageIntegratorMT::IntegrateRow, this, ref(source_channel), ref(target_channel), i, true));
		}
		for (thread& t : row_threads) t.join();
	}
    while (processed_cols < source_channel.cols) {
		std::vector<thread> col_threads;
        for (int j = processed_cols; j < min(threads_num + processed_cols, source_channel.cols); ++j) {
            col_threads.push_back(thread(&ImageIntegratorMT::IntegrateColumn, this, ref(source_channel), ref(target_channel), j, false));
		}
		for (thread& t : col_threads) t.join();
	}
}

/**
 * @brief IntegrateImageMT Многопоточное получение интегрального изображение для n-канального изображения.
 * Каналы разделяются при помощи cv::split, целевой канал приводится в соответствие с исходным по размерам и типу данных.
 * Далее каналы по очереди интегрируются и объединяются.
 * @param threads_num количество потоков
 */
void ImageIntegratorMT::IntegrateImage(int threads_num) {
    if (_image == nullptr) {
        throw std::logic_error("No source matrix");
    }
    if (_target == nullptr) {
        throw std::logic_error("No target matrix");
    }
    if (_image->empty()) {
        *_target = *_image;
        return;
    }
    size_t chans = size_t(_image->channels());
    cv::Mat conv_image;
    vector<cv::Mat> split_source(chans);
    vector<cv::Mat> split_target(chans);

    _image->convertTo(conv_image, CV_64FC(_image->channels()));
    *_target = conv_image;
	cv::split(conv_image, split_source);
    cv::split(*_target, split_target);
    for (size_t i = 0; i < chans; ++i) {
        IntegrateChannel(split_source[i], split_target[i], threads_num);
    }
    cv::merge(split_target, *_target);
}

void ExportDoubleMatTxt(cv::Mat& image, std::string filename) {
    ofstream file;
    file.open(filename);
    size_t chans = size_t(image.channels());
    vector<cv::Mat> split_image(chans);
    cv::split(image, split_image);
    for (size_t c = 0; c < chans; ++c) {
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                //file.width(8);
                //file.precision(3);
                file << split_image[c].at<double>(i,j) << ' ';
            }
            file << endl;
        }
        file << endl;
    }
    file.close();
}

void ExportDoubleMatConsole(cv::Mat& image) {
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            cout << image.at<double>(i,j) << ' ';
        }
        cout << endl;
    }
}
