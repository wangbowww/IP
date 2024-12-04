#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;

void shift(Mat plane[2]){
    // 交换一三、二四象限
    int cx = plane[0].cols / 2;
    int cy = plane[0].rows / 2; //中心点
    for(int i=0;i<2;++i){
        Mat p1(plane[i], Rect(0, 0, cx, cy)); // 左上
        Mat p2(plane[i], Rect(cx, 0, cx, cy)); // 右上
        Mat p3(plane[i], Rect(0, cy, cx, cy)); // 左下
        Mat p4(plane[i], Rect(cx, cy, cx, cy)); // 右下
        // p1 p4
        Mat temp;
        p1.copyTo(temp);
        p4.copyTo(p1);
        temp.copyTo(p4);
        // p2 p3
        p2.copyTo(temp);
        p3.copyTo(p2);
        temp.copyTo(p3);
    }
}

int main(void)
{
    Mat img = imread("E1\\E7.png", IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "Error loading image" << std::endl;
        return -1;
    }

    imshow("original", img);

    img.convertTo(img, CV_32FC1);

    Mat plane[2] = { img.clone(), Mat::zeros(img.size(), CV_32FC1) }; // 两个通道分别为实部和虚部
    Mat complex;
    merge(plane, 2, complex); // 合并成复数图像
    dft(complex, complex); // 傅立叶变换
    split(complex, plane); // 变换后分离实部和虚部

    Mat mag1, mag_log1, mag_non;
    magnitude(plane[0], plane[1], mag1);

    // 幅值对数化,便于观察
    mag1 += Scalar::all(1); // 避免log(0)
    log(mag1, mag_log1);
    normalize(mag_log1, mag_non, 0, 1, NORM_MINMAX);
    imshow("non-shifted", mag_non);

    // 移中
    shift(plane);
    Mat mag2, mag_log2, mag_shifted;
    magnitude(plane[0], plane[1], mag2);
    mag2 += Scalar::all(1);
    log(mag2, mag_log2);
    normalize(mag_log2, mag_shifted, 0, 1, NORM_MINMAX);
    imshow("shifted", mag_shifted);

    //修改频率域的图像
    Mat ans[2] = {plane[0].clone(), plane[1].clone()}; 
    for (int i = 0; i < 2; ++i)
    {
        Mat filter = Mat::ones(plane[i].size(), CV_32FC1);
        circle(filter, Point(plane[i].cols / 2, plane[i].rows / 2), 10, Scalar(0), -1); // 掩码
        multiply(plane[i], filter, ans[i]); // 应用滤波器对原图像进行修改
    }
    shift(ans);
    Mat res(complex.size(),complex.type());
    merge(ans,2,res);
    idft(res,res);
    split(res, plane);
    normalize(plane[0], plane[0], 0, 1, NORM_MINMAX);
    imshow("modified idft", plane[0]);

    // 逆变换
    idft(complex, complex);
    split(complex, plane);
    normalize(plane[0], plane[0], 0, 1, NORM_MINMAX);
    imshow("idft", plane[0]);

    waitKey(0);
    return 0;
}

