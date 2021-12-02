#include <gtest/gtest.h>
#include "../integral_image.h"

void SetIntegerMat(cv::Mat& m) {
    cv::Mat temp(3, 2, CV_32S);
    temp.at<int>(0,0) = 0;
    temp.at<int>(0,1) = 1;
    temp.at<int>(1,0) = 2;
    temp.at<int>(1,1) = 3;
    temp.at<int>(2,0) = 4;
    temp.at<int>(2,1) = 5;
    m = temp;
}

void SetDoubleMat(cv::Mat& m) {
    cv::Mat temp(2, 3, CV_64F);
    temp.at<double>(0,0) = 0.1;
    temp.at<double>(0,1) = 0.2;
    temp.at<double>(0,2) = 0.3;
    temp.at<double>(1,0) = 0.4;
    temp.at<double>(1,1) = 0.5;
    temp.at<double>(1,2) = 0.6;
    m = temp;
}

// Проверка исключений при отсутствии указателей на начальное / конечное изображение
TEST(CriticalCasesTest, TestEmptyFields) {
    cv::Mat source(2, 3, CV_64F);
    cv::Mat target(2, 3, CV_64F);
    ImageIntegratorMT integrator;
    EXPECT_THROW(integrator.IntegrateImage(2), std::logic_error);
    integrator.SetImage(source);
    EXPECT_THROW(integrator.IntegrateImage(2), std::logic_error);
    integrator.SetTarget(target);
    EXPECT_NO_THROW(integrator.IntegrateImage(2));
}

// Проверка пустой матрицы на входе
TEST(CriticalCasesTest, TestEmptyImage) {
    cv::Mat source;
    cv::Mat integral(2, 3, CV_64F);
    ImageIntegratorMT integrator;
    integrator.SetImage(source);
    integrator.SetTarget(integral);
    integrator.IntegrateImage(2);

    EXPECT_EQ(integral.rows, source.rows);
    EXPECT_EQ(integral.cols, source.cols);
}

// Целевое изображение не совпадает по размеру и типу данных, должно пройти конверсию
TEST(DoubleImageTest, TestTargetFormat) {
    int thread_num = 4;
    cv::Mat source;
    SetDoubleMat(source);
    cv::Mat integral(source.cols, source.rows, CV_32SC(source.channels()));

    ImageIntegratorMT integrator;
    integrator.SetImage(source);
    integrator.SetTarget(integral);
    integrator.IntegrateImage(thread_num);

    EXPECT_EQ(integral.rows, source.rows);
    EXPECT_EQ(integral.cols, source.cols);
    EXPECT_DOUBLE_EQ(integral.at<double>(0, 0), 0.1);
    EXPECT_DOUBLE_EQ(integral.at<double>(0, 1), 0.3);
    EXPECT_DOUBLE_EQ(integral.at<double>(0, 2), 0.6);
    EXPECT_DOUBLE_EQ(integral.at<double>(1, 0), 0.5);
    EXPECT_DOUBLE_EQ(integral.at<double>(1, 1), 1.2);
    EXPECT_DOUBLE_EQ(integral.at<double>(1, 2), 2.1);
}

// Количество потоков больше, чем измерения матрицы
TEST(DoubleImageTest, TestIntegrateImageMT) {
    int thread_num = 4;
    cv::Mat source;
    SetDoubleMat(source);
    cv::Mat integral(source.rows, source.cols, CV_64FC(source.channels()));

    ImageIntegratorMT integrator;
    integrator.SetImage(source);
    integrator.SetTarget(integral);
    integrator.IntegrateImage(thread_num);

    EXPECT_DOUBLE_EQ(integral.at<double>(0, 0), 0.1);
    EXPECT_DOUBLE_EQ(integral.at<double>(0, 1), 0.3);
    EXPECT_DOUBLE_EQ(integral.at<double>(0, 2), 0.6);
    EXPECT_DOUBLE_EQ(integral.at<double>(1, 0), 0.5);
    EXPECT_DOUBLE_EQ(integral.at<double>(1, 1), 1.2);
    EXPECT_DOUBLE_EQ(integral.at<double>(1, 2), 2.1);
}

// Использование функции напрямую на одноканальном изображении
TEST(DoubleImageTest, TestIntegrateChannelFT) {
    int thread_num = 2;
    cv::Mat source;
    SetDoubleMat(source);
    cv::Mat integral(source.rows, source.cols, CV_64FC(source.channels()));

    ImageIntegratorMT integrator;
    integrator.SetImage(source);
    integrator.SetTarget(integral);
    integrator.IntegrateChannel(source, integral, thread_num);

    EXPECT_DOUBLE_EQ(integral.at<double>(0, 0), 0.1);
    EXPECT_DOUBLE_EQ(integral.at<double>(0, 1), 0.3);
    EXPECT_DOUBLE_EQ(integral.at<double>(0, 2), 0.6);
    EXPECT_DOUBLE_EQ(integral.at<double>(1, 0), 0.5);
    EXPECT_DOUBLE_EQ(integral.at<double>(1, 1), 1.2);
    EXPECT_DOUBLE_EQ(integral.at<double>(1, 2), 2.1);
}

// Количество потоков меньше, чем измерения матрицы
TEST(IntegerImageTest, TestIntegrateImageFT) {
    int thread_num = 2;
    cv::Mat source;
    SetIntegerMat(source);
	cv::Mat integral(source.rows, source.cols, CV_64FC(source.channels()));

    ImageIntegratorMT integrator;
	integrator.SetImage(source);
	integrator.SetTarget(integral);
    integrator.IntegrateImage(thread_num);

    EXPECT_DOUBLE_EQ(integral.at<double>(0, 0), 0);
    EXPECT_DOUBLE_EQ(integral.at<double>(0, 1), 1);
    EXPECT_DOUBLE_EQ(integral.at<double>(1, 0), 2);
    EXPECT_DOUBLE_EQ(integral.at<double>(1, 1), 6);
    EXPECT_DOUBLE_EQ(integral.at<double>(2, 0), 6);
    EXPECT_DOUBLE_EQ(integral.at<double>(2, 1), 15);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
