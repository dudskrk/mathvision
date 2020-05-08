#if 1
#include <opencv2\opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>

#define WIDTH 46
#define HEIGHT 56

enum DISTANCE {
	L1 = 0,
	L2,
	MAHAL,
	MODIFIED_SSE,
};

using namespace std;
using namespace cv;

void init_vector(Mat &image) {
	string file;
	Mat temp, reshape;
	image = Mat(400, WIDTH * HEIGHT, CV_8UC1);
	for (int i = 1; i <= 40; i++) {
		for (int j = 1; j <= 10; j++) {
			file = "img/s" + to_string(i) + "_" + to_string(j) + ".png";
			temp = imread(file, 0);
			reshape = temp.reshape(1, 1);
			for (int k = 0; k < temp.rows * temp.cols; k++)
				image.at<uchar>((i - 1) * 10 + j - 1, k) = reshape.at<uchar>(0, k);
		}
	}
}

void do_PCA(Mat &src, Mat &src_360, PCA &pca, Mat &src_cov) {
	int index = 0;
	src_360 = Mat(src.rows - 40, src.cols, src.type());
	for (int i = 0; i < src.rows; i++) {
		if (i % 10 == 0)
			continue;
		else {
			for (int j = 0; j < src.cols; j++) {
				src_360.at<uchar>(index, j) = src.at<uchar>(i, j);
			}
		}
		index++;
	}
	// k = max components
	pca = PCA(src_360, Mat(), PCA::DATA_AS_ROW);
	// k = m components
	//pca = PCA(src_360, Mat(), PCA::DATA_AS_ROW, m);
}

void scaling_eigenface(Mat &src, Mat &src_scaled) {
	src_scaled = Mat(src.rows, src.cols, CV_8UC1);
	double min, max;
	minMaxLoc(src, &min, &max);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			src_scaled.at<uchar>(i, j) = (src.at<float>(i, j) - min) / (max - min) * 255;
		}
	}
}

void get_eigenface(Mat &src, Mat &src_mean, int k) {
	Mat temp, temp_scaled;
	for (int i = 0; i < k; i++) {
		temp = src.row(i).reshape(1, HEIGHT);
		scaling_eigenface(temp, temp_scaled);
		imwrite("eigenface/eigenface_360_" + to_string(i + 1) + "_" + to_string(k) + ".jpg", temp_scaled);
	}
}

double get_reconstruction_rate(Mat &src, Mat &backproj) {
	double similarity, a1 = 0, a2 = 0;
	for (int i = 0; i < src.cols; i++) {
		a1 += src.at<uchar>(0, i) * src.at<uchar>(0, i);
		a2 += backproj.at<float>(0, i) * backproj.at<float>(0, i);
	}
	similarity = src.dot(backproj) / sqrt(a1 * a2);
	return similarity;
}

// reconstructed face = mean face + sum(alpha(i) * face(i))
void get_reconstruction(Mat &image, Mat &src, Mat &src_mean, Mat &src_reconstruction, int num, int k) {
	double weight;
	Mat temp = image.row(num * 10).clone();
	temp -= src_mean;
	temp.convertTo(temp, CV_32FC1);
	src_reconstruction = src_mean.clone();
	src_reconstruction.convertTo(src_reconstruction, CV_32FC1);
	
	for (int i = 0; i < k; i++) {
		weight = temp.dot(src.row(i));
		src_reconstruction += src.row(i) * weight;
	}
	cout << get_reconstruction_rate(image.row(num * 10), src_reconstruction) << endl;
	src_reconstruction.convertTo(src_reconstruction, CV_8UC1);
	imwrite("eigenface/reconstruction_" + to_string(num + 1) + "_" + to_string(k) + ".jpg", src_reconstruction.reshape(1, HEIGHT));
}

// a = test, b = train, k = vector num, num = distance mode
double get_distance(Mat &a, Mat &b, int k, int num) {
	double *distance = new double[b.rows];
	memset(distance, 0, sizeof(double) * b.rows);
	double min = 100000000000, index, a1, a2;
	if (num == DISTANCE::L2) {
		for (int i = 0; i < b.rows; i++) {
			if (i % 10 == 0) // all of test data
				continue;
			for (int j = 0; j < k; j++) {
				a1 = a.at<float>(0, j);
				a2 = b.at<float>(i, j);
				distance[i] += sqrt(pow(a1 - a2, 2));
			}
		}
	}
	else if (num == DISTANCE::MAHAL) {

	}
	else if (num == DISTANCE::MODIFIED_SSE) {

	}
	// min value & index
	for (int i = 0; i < b.rows; i++) {
		if (distance[i] < min && i % 10 != 0) {
			min = distance[i];
			index = i;
		}
	}
	delete distance;
	return index;
}

void classifier(Mat &src, PCA &pca_360, int k, int num) {
	int index, count = 0;
	double thresh = 0.9;
	Mat projection = pca_360.project(src);
	for (int i = 0; i < 40; i++) {
		Mat train = projection.row(i * 10).clone();
		index = get_distance(train, projection, k, num);
		if (index / 10 == i)
			count++;
	}
	cout << k << "-space : " << count / 40.0 << endl;
}

void classifier_one(Mat &train, Mat &test, PCA &pca_360, int k, int num) {
	int index;
	double reconstruction_rate;
	Mat projection_test = pca_360.project(test);
	Mat projection_train = pca_360.project(train);
	Mat backprojection_test = pca_360.backProject(projection_test);
	index = get_distance(projection_test, projection_train, k, num);
	reconstruction_rate = get_reconstruction_rate(test, backprojection_test);
	cout << "similarity between test and backprojection : " << reconstruction_rate << endl;
	imshow("similar index = " + to_string(index), train.row(index).reshape(1, HEIGHT));
	imshow("src index = " + to_string(index), test.reshape(1, HEIGHT));
}

double covariance(Mat &x, Mat &y) {
	double covariance = 0, mean_x = 0, mean_y = 0;
	int n = x.cols;
	for (int i = 0; i < n; i++) {
		mean_x += x.at<uchar>(0, i);
		mean_y += y.at<uchar>(0, i);
	}
	mean_x /= (double)x.cols;	mean_y /= (double)y.cols;
	for (int i = 0; i < n; i++) {
		covariance += (x.at<uchar>(0, i) - mean_x) * (y.at<uchar>(0, i) - mean_y);
	}
	return covariance / (double)(n - 1);
}

void get_covariance_matrix(Mat &src, Mat &src_cov) {
	int n = src.cols;
	Mat x, y;
	src_cov = Mat(n, n, CV_8UC1);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			x = src.col(i).clone();
			y = src.col(j).clone();
			src_cov.at<uchar>(i, j) = covariance(x, y);
		}
	}
}

int main() {
	Mat images, vectors_360, vectors_mean, vectors_cov, vectors_evector, vectors_evalue;
	Mat reconstruction_1, reconstruction_10, reconstruction_100, reconstruction_200;
	Mat cv_eigenvector, cv_eigenvalue, cv_mean;
	PCA pca;
	int k = 10;

	cout << "initialize vectors" << endl;
	init_vector(images);

	cout << "PCA...." << endl;
	do_PCA(images, vectors_360, pca, vectors_cov);
	vectors_evalue = pca.eigenvalues.clone();
	vectors_evector = pca.eigenvectors.clone();
	vectors_mean = pca.mean.clone();

	cout << "(1) get scaled eigen face k = " + to_string(k) << endl;
	get_eigenface(vectors_evector, vectors_mean, k);
	cout << "(1) done" << endl;
	
	//calcCovarMatrix(vectors_360, vectors_cov, vectors_mean, CV_COVAR_COLS | CV_COVAR_NORMAL | CV_COVAR_SCALE);
	cout << "(2) get reconstruction image 28th person with k = 1, 10, 100, 200" << endl;
	get_reconstruction(images, vectors_evector, vectors_mean, reconstruction_1, 27, 1);
	get_reconstruction(images, vectors_evector, vectors_mean, reconstruction_10, 27, 10);
	get_reconstruction(images, vectors_evector, vectors_mean, reconstruction_100, 27, 100);
	get_reconstruction(images, vectors_evector, vectors_mean, reconstruction_200, 27, 200);
	cout << "(2) done" << endl;

	cout << "(3) classify between test data and train data" << endl;
	classifier(images, pca, 1, DISTANCE::L2);
	classifier(images, pca, 10, DISTANCE::L2);
	classifier(images, pca, 100, DISTANCE::L2);
	classifier(images, pca, 200, DISTANCE::L2);
	cout << "(3) done" << endl;

	cout << "(4) similarity between me and train data" << endl;
	Mat me = imread("img/me.jpg", 0);
	Mat me_glasses = imread("img/me_glasses.jpg", 0);
	classifier_one(images, me.reshape(1, 1), pca, 200, DISTANCE::L2);
	classifier_one(images, me_glasses.reshape(1, 1), pca, 200, DISTANCE::L2);
	cout << "(4) done" << endl;
	waitKey(0);
	destroyAllWindows();

	return 0;
}
#endif