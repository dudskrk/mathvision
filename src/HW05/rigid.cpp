#include <iostream>
#include <vector>
#include <opencv2\opencv.hpp>
#include <cmath>

using namespace std;
using namespace cv;

void getRotationTheta(Mat &rotation, Point3d axis, double theta) {
	double rad = theta * CV_PI / 180.0;
	double unit_axis_value = sqrt(pow(axis.x, 2) + pow(axis.y, 2) + pow(axis.z, 2));
	Point3d unit_axis = Point3d(axis.x / unit_axis_value, axis.y / unit_axis_value, axis.z / unit_axis_value);
	
	rotation.at<double>(0, 0) = cos(rad) + pow(unit_axis.x, 2) * (1 - cos(rad));
	rotation.at<double>(1, 0) = unit_axis.x * unit_axis.y * (1 - cos(rad)) - unit_axis.z * sin(rad);
	rotation.at<double>(2, 0) = unit_axis.x * unit_axis.z * (1 - cos(rad)) + unit_axis.y * sin(rad);
	rotation.at<double>(0, 1) = unit_axis.y * unit_axis.x * (1 - cos(rad)) + unit_axis.z * sin(rad);
	rotation.at<double>(1, 1) = cos(rad) + pow(unit_axis.y, 2) * (1 - cos(rad));
	rotation.at<double>(2, 1) = unit_axis.y * unit_axis.z * (1 - cos(rad)) - unit_axis.x * sin(rad);
	rotation.at<double>(0, 2) = unit_axis.z * unit_axis.x * (1 - cos(rad)) - unit_axis.y * sin(rad);
	rotation.at<double>(1, 2) = unit_axis.z * unit_axis.y * (1 - cos(rad)) + unit_axis.x * sin(rad);
	rotation.at<double>(2, 2) = cos(rad) + pow(unit_axis.z, 2) * (1 - cos(rad));
}

void getCrossProduct(Point3d &cross, Point3d &v1, Point3d &v2) {
	float a1 = v1.x; float a2 = v1.y; float a3 = v1.z;
	float b1 = v2.x; float b2 = v2.y; float b3 = v2.z;

	cross.x = a2 * b3 - a3 * b2;
	cross.y = a3 * b1 - a1 * b3;
	cross.z = a1 * b2 - a2 * b1;
}

void getNormalVector(Point3d &normal, vector<Point3d> &points) {
	float a1 = points[0].x; float a2 = points[0].y; float a3 = points[0].z;
	float b1 = points[1].x; float b2 = points[1].y; float b3 = points[1].z;
	float c1 = points[2].x; float c2 = points[2].y; float c3 = points[2].z;

	getCrossProduct(normal, points[1] - points[0], points[2] - points[0]);
}

double getAngle(Point3d &v1, Point3d &v2) {
	double angle;
	angle = acos((v1.x * v2.x + v1.y * v2.y + v1.z * v2.z)
		/ (sqrt(pow(v1.x, 2) + pow(v1.y, 2) + pow(v1.z, 2))
			* sqrt(pow(v2.x, 2) + pow(v2.y, 2) + pow(v2.z, 2))));

	angle *= 180.0 / CV_PI;
	return min(angle, 180 - angle);
}

void getMultiply(Point3d &result, Mat &matrix, Point3d &point) {
	result.x = matrix.at<double>(0, 0) * point.x + matrix.at<double>(1, 0) * point.y + matrix.at<double>(2, 0) * point.z;
	result.y = matrix.at<double>(0, 1) * point.x + matrix.at<double>(1, 1) * point.y + matrix.at<double>(2, 1) * point.z;
	result.z = matrix.at<double>(0, 2) * point.x + matrix.at<double>(1, 2) * point.y + matrix.at<double>(2, 2) * point.z;
}

int main() {
	Mat translation(4, 4, CV_64FC1);
	Mat rotation_theta1(3, 3, CV_64FC1);
	Mat rotation_theta2(3, 3, CV_64FC1);
	vector<Point3d> p, q;
	Point3d p1, p2, p3, p4, p5, q1, q2, q3, q4, q5, result1, result2;
	Point3d p_normal, q_normal, pq, rpq, rp;
	float angle1, angle2;

	p1 = Point3d(-0.5, 0, 2.121320);
	p2 = Point3d(0.5, 0, 2.121320);
	p3 = Point3d(0.5, -0.707107, 2.828427);
	p4 = Point3d(0.5, 0.707107, 2.828427);
	p5 = Point3d(1, 1, 1);
	p.push_back(p1); p.push_back(p2); p.push_back(p3); p.push_back(p4); p.push_back(p5);

	q1 = Point3d(1.363005, -0.427130, 2.339082);
	q2 = Point3d(1.748084, 0.437983, 2.017688);
	q3 = Point3d(2.636561, 0.184843, 2.400710);
	q4 = Point3d(1.4981, 0.8710, 2.8837);
	q.push_back(q1); q.push_back(q2); q.push_back(q3); q.push_back(q4);

	getNormalVector(p_normal, p); 
	getNormalVector(q_normal, q); 
	angle1 = getAngle(p_normal, q_normal); 
	getCrossProduct(pq, p_normal, q_normal);
	getRotationTheta(rotation_theta1, pq, angle1); 
	getMultiply(rp, rotation_theta1, p3 - p1); 
	angle2 = getAngle(rp, q3 - q1);
	getRotationTheta(rotation_theta2, q_normal, angle2);

	Mat rotation_theta2t = rotation_theta2.t();
	getMultiply(result1, rotation_theta1, p4 - p1);
	getMultiply(result2, rotation_theta2t, result1);
	q4 = result2 + q1;

	getMultiply(result1, rotation_theta1, p5 - p1);
	getMultiply(result2, rotation_theta2t, result1);
	q5 = result2 + q1;
	
	return 0;
}