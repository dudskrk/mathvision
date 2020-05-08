#include "polygon_demo.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

PolygonDemo::PolygonDemo()
{
	m_data_ready = false;
}

PolygonDemo::~PolygonDemo()
{
}

void PolygonDemo::refreshWindow()
{
	Mat frame = Mat::zeros(480, 640, CV_8UC3);
	if (!m_data_ready)
		putText(frame, "Input data points (double click: finish)", Point(10, 470), FONT_HERSHEY_SIMPLEX, .5, Scalar(0, 148, 0), 1);

	drawPolygon(frame, m_data_pts, m_data_ready);
	if (m_data_ready)
	{
		// polygon area
		if (m_param.compute_area)
		{
			int area = polyArea(m_data_pts);
			char str[100];
			sprintf_s(str, 100, "Area = %d", area);
			putText(frame, str, Point(25, 25), FONT_HERSHEY_SIMPLEX, .8, Scalar(0, 255, 255), 1);
		}

		// pt in polygon
		if (m_param.check_ptInPoly)
		{
			for (int i = 0; i < (int)m_test_pts.size(); i++)
			{
				if (ptInPolygon(m_data_pts, m_test_pts[i]))
				{
					circle(frame, m_test_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
				}
				else
				{
					circle(frame, m_test_pts[i], 2, Scalar(128, 128, 128), CV_FILLED);
				}
			}
		}

		// homography check
		if (m_param.check_homography && m_data_pts.size() == 4)
		{
			// rect points
			int rect_sz = 100;
			vector<Point> rc_pts;
			rc_pts.push_back(Point(0, 0));
			rc_pts.push_back(Point(0, rect_sz));
			rc_pts.push_back(Point(rect_sz, rect_sz));
			rc_pts.push_back(Point(rect_sz, 0));
			rectangle(frame, Rect(0, 0, rect_sz, rect_sz), Scalar(255, 255, 255), 1);

			// draw mapping
			char* abcd[4] = { "A", "B", "C", "D" };
			for (int i = 0; i < 4; i++)
			{
				line(frame, rc_pts[i], m_data_pts[i], Scalar(255, 0, 0), 1);
				circle(frame, rc_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
				circle(frame, m_data_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
				putText(frame, abcd[i], m_data_pts[i], FONT_HERSHEY_SIMPLEX, .8, Scalar(0, 255, 255), 1);
			}

			// check homography
			int homo_type = classifyHomography(rc_pts, m_data_pts);
			char type_str[100];
			switch (homo_type)
			{
			case NORMAL:
				sprintf_s(type_str, 100, "normal");
				break;
			case CONCAVE:
				sprintf_s(type_str, 100, "concave");
				break;
			case TWIST:
				sprintf_s(type_str, 100, "twist");
				break;
			case REFLECTION:
				sprintf_s(type_str, 100, "reflection");
				break;
			case CONCAVE_REFLECTION:
				sprintf_s(type_str, 100, "concave reflection");
				break;
			}

			putText(frame, type_str, Point(15, 125), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 1);
		}

		// fit circle
		if (m_param.fit_circle)
		{
			Point2d center;
			double radius = 0;
			bool ok = fitCircle(m_data_pts, center, radius);
			if (ok)
			{
				circle(frame, center, (int)(radius + 0.5), Scalar(0, 255, 0), 1);
				circle(frame, center, 2, Scalar(0, 255, 0), CV_FILLED);
			}
		}

		// fit ellipse
		if (m_param.fit_ellipse)
		{
			Point2d center;
			Size radius;
			double theta;
			bool ok = fitEllipse(m_data_pts, center, radius, theta);
			if (ok)
			{
				std::cout << center << std::endl;
				std::cout << radius << std::endl;
				ellipse(frame, center, radius, theta, 0, 360, Scalar(0, 255, 0));
				circle(frame, center, 2, Scalar(0, 255, 0), CV_FILLED);
			}
		}
	}
	imshow("PolygonDemo", frame);
}

// return the area of polygon
int PolygonDemo::polyArea(const std::vector<cv::Point>& vtx)
{
	return 0;
}

// return true if pt is interior point
bool PolygonDemo::ptInPolygon(const std::vector<cv::Point>& vtx, Point pt)
{
	return false;
}

// return homography type: NORMAL, CONCAVE, TWIST, REFLECTION, CONCAVE_REFLECTION
int PolygonDemo::classifyHomography(const std::vector<cv::Point>& pts1, const std::vector<cv::Point>& pts2)
{
	if (pts1.size() != 4 || pts2.size() != 4) return -1;

	return NORMAL;
}

// estimate a circle that best approximates the input points and return center and radius of the estimate circle
bool PolygonDemo::fitCircle(const std::vector<cv::Point>& pts, cv::Point2d& center, double& radius)
{
	int n = (int)pts.size();
	double a, b, c; // c = a^2 + b^2 - r^2
	if (n < 3) return false;

	cv::Mat A = cv::Mat(pts.size(), 3, CV_64FC1);
	cv::Mat B = cv::Mat(3, 1, CV_64FC1);
	cv::Mat C = cv::Mat(pts.size(), 1, CV_64FC1);

	// Initialize
	for (int i = 0; i < pts.size(); i++){
		A.at<double>(i, 0) = 2 * pts[i].x;
		A.at<double>(i, 1) = 2 * pts[i].y;
		A.at<double>(i, 2) = -1;
		C.at<double>(i, 0) = pts[i].x * pts[i].x + pts[i].y * pts[i].y;
	}

	// pseudo inverse
	cv::Mat pinvA = (A.t() * A).inv() * A.t();
	B = pinvA * C;
	a = B.at<double>(0, 0);
	b = B.at<double>(1, 0);
	c = B.at<double>(2, 0);
	radius = sqrt(a * a + b * b - c);
	center = Point2d(a, b);

	std::cout << A << std::endl;
	std::cout << C << std::endl;
	std::cout << pinvA << std::endl;
	std::cout << B << std::endl;

	return true;
}

bool PolygonDemo::fitEllipse(const std::vector<cv::Point>& pts, cv::Point2d& center, cv::Size &radius, double &angle)
{
	int n = (int)pts.size();
	//if (n < 6) return false;

	double a, b, c, d, e, f, w, h, x0, y0;
	cv::Mat A = cv::Mat(pts.size(), 6, CV_64FC1);

	// Initialize
	for (int i = 0; i < pts.size(); i++){
		A.at<double>(i, 0) = pts[i].x * pts[i].x;
		A.at<double>(i, 1) = pts[i].x * pts[i].y;
		A.at<double>(i, 2) = pts[i].y * pts[i].y;
		A.at<double>(i, 3) = pts[i].x;
		A.at<double>(i, 4) = pts[i].y;
		A.at<double>(i, 5) = 1;
	}

	cv::Mat x, u, s, vt;
	cv::SVDecomp(A, u, s, vt);

	x = vt.row(vt.rows - 1);
	a = x.at<double>(0, 0);
	b = x.at<double>(0, 1);
	c = x.at<double>(0, 2);
	d = x.at<double>(0, 3);
	e = x.at<double>(0, 4);
	f = x.at<double>(0, 5);

	std::cout << A << std::endl;
	std::cout << u << std::endl;
	std::cout << s << std::endl;
	std::cout << vt << std::endl;
	std::cout << A * x.t() << std::endl;

	w = -sqrt(2 * (a * e * e + c * d * d - b * d * e + (b * b - 4 * a * c) * f) *
		((a + c) + sqrt((a - c) * (a - c) + b * b)))
		/ (b * b - 4 * a * c);
	h = -sqrt(2 * (a * e * e + c * d * d - b * d * e + (b * b - 4 * a * c) * f) *
		((a + c) - sqrt((a - c) * (a - c) + b * b)))
		/ (b * b - 4 * a * c);
	x0 = (2 * c * d - b * e) / (b * b - 4 * a * c);
	y0 = (2 * a * e - b * d) / (b * b - 4 * a * c);

	if (b != 0)
		angle = atan((c - a - sqrt((a - c) * (a - c) + b * b)) / b) * 180 / CV_PI;
	else if (b == 0 && a < c)
		angle = 0;
	else if (b == 0 && a > c)
		angle = 90;
	
	center = Point2d(x0, y0);
	radius = Size(w, h);
	return true;
}

void PolygonDemo::drawPolygon(Mat& frame, const std::vector<cv::Point>& vtx, bool closed)
{
	int i = 0;
	for (i = 0; i < (int)m_data_pts.size(); i++)
	{
		circle(frame, m_data_pts[i], 2, Scalar(255, 255, 255), CV_FILLED);
	}
	if (m_param.draw_line) {
		for (i = 0; i < (int)m_data_pts.size() - 1; i++)
		{
			line(frame, m_data_pts[i], m_data_pts[i + 1], Scalar(255, 255, 255), 1);
		}
		if (closed)
		{
			line(frame, m_data_pts[i], m_data_pts[0], Scalar(255, 255, 255), 1);
		}
	}
}

void PolygonDemo::handleMouseEvent(int evt, int x, int y, int flags)
{
	if (evt == CV_EVENT_LBUTTONDOWN)
	{
		std::printf("%d %d\n", x, y);
		if (!m_data_ready)
		{
			m_data_pts.push_back(Point(x, y));
		}
		else
		{
			m_test_pts.push_back(Point(x, y));
		}
		refreshWindow();
	}
	else if (evt == CV_EVENT_LBUTTONUP)
	{
	}
	else if (evt == CV_EVENT_LBUTTONDBLCLK)
	{
		m_data_ready = true;
		refreshWindow();
	}
	else if (evt == CV_EVENT_RBUTTONDBLCLK)
	{
	}
	else if (evt == CV_EVENT_MOUSEMOVE)
	{
	}
	else if (evt == CV_EVENT_RBUTTONDOWN)
	{
		m_data_pts.clear();
		m_test_pts.clear();
		m_data_ready = false;
		refreshWindow();
	}
	else if (evt == CV_EVENT_RBUTTONUP)
	{
	}
	else if (evt == CV_EVENT_MBUTTONDOWN)
	{
	}
	else if (evt == CV_EVENT_MBUTTONUP)
	{
	}

	if (flags&CV_EVENT_FLAG_LBUTTON)
	{
	}
	if (flags&CV_EVENT_FLAG_RBUTTON)
	{
	}
	if (flags&CV_EVENT_FLAG_MBUTTON)
	{
	}
	if (flags&CV_EVENT_FLAG_CTRLKEY)
	{
	}
	if (flags&CV_EVENT_FLAG_SHIFTKEY)
	{
	}
	if (flags&CV_EVENT_FLAG_ALTKEY)
	{
	}
}