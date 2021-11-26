#include <cstring>
#include <iostream>
#include "integral_image.h"
using namespace std;

int main(int argc, char* argv[]) {
    int threads_num = 0;
    vector<string> image_input;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-i") == 0) {
            ++i;
            while (i < argc && strcmp(argv[i], "-t") != 0) {
                image_input.push_back(argv[i]);
                ++i;
            }
        }
        if (i < argc && strcmp(argv[i],"-t") == 0) {
            ++i;
            if (i < argc) {
                threads_num = *argv[i] - '0';
            }
            else {
                cerr << "ERROR: Thread number not specified." << endl;
                return 1;
            }
            if (threads_num < 0 || size_t(threads_num) > std::thread::hardware_concurrency()) {
                cerr << "ERROR: Impossible number of threads." << endl;
                return 2;
            }
        }
    }

    if (threads_num == 0) threads_num = int(std::thread::hardware_concurrency());
    for (string image_path : image_input) {
        cv::Mat source = cv::imread(image_path, cv::IMREAD_UNCHANGED);
        if (!source.data) {
            cerr << "ERROR: Unable to read image: " << image_path << endl;
        }
        else {
            cv::Mat integral(source.rows, source.cols, CV_64FC(source.channels()));
            ImageIntegratorMT integrator;
            integrator.SetImage(source);
            integrator.SetTarget(integral);
            integrator.IntegrateImage(threads_num);
            string outfile = image_path;
            outfile.append(".integral.txt");
            ExportDoubleMatTxt(integral, outfile);
        }
        
    }
    return 0;
}
