#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
// 计算颜色差异
inline double colorDiff(const Vec3b& a, const Vec3b& b)
{
    double res = 0;
    for (int i = 0; i < 3; ++i) {
        int t = static_cast<int>(a[i]) - static_cast<int>(b[i]); // 防止负数问题
        res += t * t;
    }
    return res;
}

void bilateralFilter(Mat& src, Mat& dst, int d, double sigmad, double sigmar)
{
    assert(d % 2 == 1 && d >= 1);
    const int r = d / 2;
    const double squarer = sigmar * sigmar;
    const double squared = sigmad * sigmad;

    // 计算Gauss空间weight
    std::vector<std::vector<double>> dw(d, std::vector<double>(d,0));
    double sum = 0;
    for (int i = -r; i <= r; ++i) {
        for (int j = -r; j <= r; ++j) {
            double w = exp(-(i * i + j * j) / (2 * squared));
            dw[i + r][j + r] = w;
            sum += w;
        }
    }
    // 归一化
    for (int i = -r; i <= r; ++i) {
        for (int j = -r; j <= r; ++j) {
            dw[i + r][j + r] /= sum;
        }
    }
    dst.create(src.size(), src.type());

    for (int y = 0; y < src.rows; ++y) {
        Vec3b* ptr_dst = dst.ptr<Vec3b>(y);
        const Vec3b* ptr_src = src.ptr<Vec3b>(y);
        #pragma omp parallel for
        for (int x = 0; x < src.cols; ++x) {
            Vec3b cur = ptr_src[x];
            double sum = 0;
            double rr = 0, gg = 0, bb = 0;

            for (int i = -r; i <= r; ++i) {
                for (int j = -r; j <= r; ++j) {
                    if (x + j >= 0 && x + j < src.cols && y + i >= 0 && y + i < src.rows) {
                        const Vec3b& pixel = src.at<Vec3b>(y + i, x + j);
                        double w1 = dw[i + r][j + r];
                        double w2 = exp(-colorDiff(cur, pixel) / (2 * squarer));
                        w2 = 1;
                        double w = w1 * w2;
                        rr += pixel[0] * w;
                        gg += pixel[1] * w;
                        bb += pixel[2] * w;
                        sum += w;
                    }
                }
            }
            if (sum == 0) sum = 1;
            ptr_dst[x][0] = static_cast<uchar>(rr / sum);
            ptr_dst[x][1] = static_cast<uchar>(gg / sum);
            ptr_dst[x][2] = static_cast<uchar>(bb / sum);
        }
    }
}

int main()
{
    Mat img = imread("E1\\test-data\\test-jpg.jpg");
    if (img.empty()) {
        std::cout << "Error loading image" << std::endl;
        return -1;
    }

    Mat ans;
    
    TickMeter tm;
    double tim1=0,tim2=0;
    for(int i=0;i<1;++i){
        tm.reset();
        tm.start();
        bilateralFilter(img, ans, 9, 10, 100);
        tm.stop();
        tim1 += tm.getTimeMilli();
        tm.reset();
        tm.start();
        cv::bilateralFilter(img, ans, 9, 10, 100);
        tm.stop();
        tim2 += tm.getTimeMilli();
    }
    printf("%.2lf\n%.2lf\n", tim1 / 10, tim2 / 10);
    imshow("Original", img);
    imshow("My Filter", ans);
    waitKey();

    return 0;
}