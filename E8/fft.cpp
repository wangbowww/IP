#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;

void myMatchTemplate(const Mat &img, const Mat &templ, Mat &ans){
    assert(img.channels() == 1 && templ.type() == img.type());
    int rows = img.rows, cols = img.cols;
    int tr = templ.rows, tc = templ.cols;
    if(rows - tr + 1 != ans.rows || cols - tc + 1 != ans.cols || ans.type() != CV_32FC1){
        ans.create(rows - tr + 1, cols - tc + 1, CV_32FC1);
    }
    // 开始匹配
    float *p = ans.ptr<float>();
    for (int y = 0; y < rows - tr + 1; ++y) {
        for (int x = 0; x < cols - tc + 1; ++x) {
            float dis = 0; // 差异值
            for(int i = 0; i < tr; ++i){
                for(int j = 0; j < tc; ++j){
                    int t1 = img.at<uchar>(y + i, x + j);
                    int t2 = templ.at<uchar>(i, j);
                    int t3 = t1 - t2;
                    dis += t3 * t3;
                }
            }
            *p = dis;
            p++;
        }
    }
}

void modifiedMatchTemplate(const Mat& img, const Mat& templ, Mat& ans)
{
    assert(img.channels() == 1 && templ.type() == img.type());
    int rows = img.rows, cols = img.cols;
    int tr = templ.rows, tc = templ.cols;
    if (rows - tr + 1 != ans.rows || cols - tc + 1 != ans.cols || ans.type() != CV_32FC1) {
        ans.create(rows - tr + 1, cols - tc + 1, CV_32FC1);
    }

    // 平方积分图
    auto createSquareIntegral = [&](const Mat& img, Mat& integral) {
        const uchar* row1 = img.ptr<uchar>();
        float* row2 = integral.ptr<float>();
        for (int y = 0; y < img.rows; ++y, row1 += img.step, row2 += integral.step / 4) {
            const uchar* p1 = row1;
            float* p2 = row2;
            for (int x = 0; x < img.cols; ++x, p1 ++, p2 ++) {
                // sum(x,y) = sum(x-1,y) + sum(x,y-1) - sum(x-1,y-1) + a[x][y];
                float t = *p1;
                *p2 = t * t;
                if (y > 0){
                    t = *(p2 - integral.step / 4);
                    *p2 += t;
                }
                if (x > 0){
                    t = *(p2 - 1);
                    *p2 += t;
                }
                if (x > 0 && y > 0){
                    t = *(p2 - integral.step / 4 - 1);
                    *p2 -= t;
                }
            }
        }
    };
    // 创建并建立平方积分图加速第三项
    Mat integral(img.size(), CV_32FC1);
    createSquareIntegral(img, integral);

    float I = 0;  // 模板图像的像素平方和(第一项)
    const uchar *p = templ.ptr<uchar>();
    for (int i = 0; i < templ.rows; ++i) {
        for (int j = 0; j < templ.cols; ++j) {
            float t = *p;
            I += t * t;
            p++;
        }
    }
    // fft处理卷积项(第二项)
    auto FFT = [&](const Mat& img, const Mat& templ, Mat& result) {
        int dft_h = cv::getOptimalDFTSize(img.rows + templ.rows - 1);
        int dft_w = cv::getOptimalDFTSize(img.cols + templ.cols - 1);

        Mat dft_img = Mat::zeros(dft_h, dft_w, CV_32FC1);
        Mat dft_templ = Mat::zeros(dft_h, dft_w, CV_32FC1);

        // 将img和templ分别拷贝放到dft_img和dft_templ的中心
        int cx = (dft_h - img.rows) / 2;
        int cy = (dft_w - img.cols) / 2;
        Mat dft_img_part = dft_img(cv::Rect(cy, cx, img.cols, img.rows));
        Mat dft_templ_part = dft_templ(cv::Rect(cy, cx, templ.cols, templ.rows));

        img.convertTo(dft_img_part, CV_32FC1);
        templ.convertTo(dft_templ_part, CV_32FC1);

        // 对dft_img和dft_templ进行DFT
        dft(dft_img, dft_img);
        dft(dft_templ, dft_templ);

        // 将DFT结果相乘
        mulSpectrums(dft_img, dft_templ, dft_img, 0, true);

        // 对乘积进行逆DFT
        dft(dft_img, dft_img, DFT_INVERSE + DFT_SCALE);

        // 提取结果的低频部分
        int result_h = img.rows - templ.rows + 1;
        int result_w = img.cols - templ.cols + 1;
        result.create(result_h, result_w, CV_32FC1);
        Mat corr = dft_img(cv::Rect(0, 0, result_w, result_h));
        corr.copyTo(result);
    };
    Mat result(ans.size(), ans.type());
    FFT(img, templ, result);
    // 计算最终结果
    for (int i = 0; i < ans.rows; ++i) {
        for (int j = 0; j < ans.cols; ++j) {
            float val = -2 * result.at<float>(i, j) + I;
            if (j > 0)
                val -= integral.at<float>(i + templ.rows - 1, j - 1);
            if (i > 0)
                val -= integral.at<float>(i - 1, j + templ.cols - 1);
            if (i > 0 && j > 0)
                val += integral.at<float>(i - 1, j - 1);
            val += integral.at<float>(i + templ.rows - 1, j + templ.cols - 1);
            ans.at<float>(i, j) = val;
        }
    }
}

int main()
{
    // 读取输入图像和模板图像
    Mat img = imread("E1\\E8-1.png", IMREAD_GRAYSCALE); // 确保是灰度图
    Mat templ = imread("E1\\E8-3.png", IMREAD_GRAYSCALE);

    if (img.empty() || templ.empty()) {
        std::cout << "Error loading image" << std::endl;
        return -1;
    }

    Mat ans;
    ans.create(img.rows - templ.rows + 1, img.cols - templ.cols + 1, CV_32FC1);

    // 时间测试
    do{
        TickMeter tm;
        double tim1 = 0, tim2 = 0, tim3 = 0;
        for (int i = 0; i < 10; ++i) {
            tm.reset();
            tm.start();
            myMatchTemplate(img, templ, ans);
            tm.stop();
            tim1 += tm.getTimeMilli();

            tm.reset();
            tm.start();
            modifiedMatchTemplate(img, templ, ans);
            tm.stop();
            tim2 += tm.getTimeMilli();

            tm.reset();
            tm.start();
            matchTemplate(img, templ, ans, TM_SQDIFF);
            tm.stop();
            tim3 += tm.getTimeMilli();
        }
        printf("mine: %.2lf\nmodified: %.2lf\nCV: %.2lf\n", tim1 / 10, tim2 / 10, tim3 / 10);
    }while(0);

    // 匹配正确性测试
    //myMatchTemplate(img, templ, ans);
    matchTemplate(img, templ, ans, TM_SQDIFF);
    //modifiedMatchTemplate(img, templ, ans);

    // 找到最匹配的位置
    Point res;
    minMaxLoc(ans, nullptr, nullptr, &res, nullptr, Mat());


    // 框出匹配区域
    rectangle(img, res, Point(res.x + templ.cols, res.y + templ.rows), Scalar::all(0), 2);

    imshow("Matched Result", img);
    waitKey(0);

    return 0;
}