#include <opencv2\opencv.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
	Mat src = imread("hw11_sample.png", -1);
	Mat src_binary_otsu, src_binary_thresh;

	int thresh = threshold(src, src_binary_otsu, 0, 255, CV_THRESH_OTSU);
	threshold(src, src_binary_thresh, 127, 255, CV_THRESH_BINARY);
	imshow("src otsu = " + to_string(thresh), src_binary_otsu);
	imshow("src thresh = 127", src_binary_thresh);
	waitKey(0);

	Mat A, X, Y;
	A = Mat(src.rows * src.cols, 6, CV_32FC1);
	Y = Mat(src.rows * src.cols, 1, CV_32FC1);
	X = Mat(6, 1, CV_32FC1);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			A.at<float>(i * src.rows + j, 0) = i * i;
			A.at<float>(i * src.rows + j, 1) = j * j;
			A.at<float>(i * src.rows + j, 2) = i * j;
			A.at<float>(i * src.rows + j, 3) = i;
			A.at<float>(i * src.rows + j, 4) = j;
			A.at<float>(i * src.rows + j, 5) = 1;
			Y.at<float>(i * src.rows + j, 0) = src.at<uchar>(i, j);
		}
	}
	// X = pinvA * Y
	float a, b, c, d, e, f;
	Mat pinvA;
	invert(A, pinvA, DECOMP_SVD);
	X = pinvA * Y;
	cout << X << ", " << X.type();

	a = X.at<float>(0, 0); b = X.at<float>(1, 0); c = X.at<float>(2, 0);
	d = X.at<float>(3, 0); e = X.at<float>(4, 0); f = X.at<float>(5, 0);

	// get 2nd order polynomial surface
	Mat LS_surface = Mat(src.rows, src.cols, CV_32FC1);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			LS_surface.at<float>(i, j) = a * i * i + b * j * j + c * i * j +
				d * i + e * j + f;
		}
	}
	LS_surface.convertTo(LS_surface, CV_8UC1);
	imshow("LS_surface", LS_surface);
	waitKey(0);

	// get threshold of 2nd order polynomial surface
	Mat result, result_binary;
	absdiff(src, LS_surface, result);
	threshold(result, result_binary, 0, 255, CV_THRESH_OTSU);
	imshow("result(src - LS_surface)", result);
	imshow("result(src - LS_surface) binary", result_binary);
	waitKey(0);

	return 0;
}