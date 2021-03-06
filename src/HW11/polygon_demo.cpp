#include "polygon_demo.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

PolygonDemo::PolygonDemo()
{
    m_data_ready = false;
	wheel_flag = 0;
}

PolygonDemo::~PolygonDemo()
{
}

void PolygonDemo::refreshWindow()
{
    Mat frame = Mat::zeros(480, 640, CV_8UC3);
    if (!m_data_ready)
		cv::putText(frame, "Input data points (double click: finish)", Point(10, 470), FONT_HERSHEY_SIMPLEX, .5, Scalar(0, 148, 0), 1);

    drawPolygon(frame, m_data_pts, m_data_ready);
    if (m_data_ready)
    {
        // polygon area
        if (m_param.compute_area)
        {
            int area = polyArea(m_data_pts);
            char str[100];
            sprintf_s(str, 100, "Area = %d", area);
			cv::putText(frame, str, Point(25, 25), FONT_HERSHEY_SIMPLEX, .8, Scalar(0, 255, 255), 1);
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
				cv::line(frame, rc_pts[i], m_data_pts[i], Scalar(255, 0, 0), 1);
				cv::circle(frame, rc_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
                cv::circle(frame, m_data_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
				cv::putText(frame, abcd[i], m_data_pts[i], FONT_HERSHEY_SIMPLEX, .8, Scalar(0, 255, 255), 1);
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

            cv::putText(frame, type_str, Point(15, 125), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 1);
        }

        // fit circle
        if (m_param.fit_circle)
        {
            Point2d center;
            double radius = 0;
            bool ok = fitCircle(m_data_pts, center, radius);
            if (ok)
            {
                circle(frame, center, (int)(radius + 0.5), Scalar(255, 0, 0), 1);
                circle(frame, center, 2, Scalar(255, 0, 0), CV_FILLED);
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
				ellipse(frame, center, radius, theta, 0, 360, Scalar(0, 255, 0));
				circle(frame, center, 2, Scalar(0, 255, 0), CV_FILLED);
			}
		}

		// fit line
		if (m_param.fit_line){
			Vec4i green, red;
			string gstr, rstr;
			bool green_ok = fitGreenLine(m_data_pts, green, gstr);
			if (green_ok)
			{
				cv::line(frame, Point(green[0], green[1]), Point(green[2], green[3]), Scalar(0, 255, 0), 1, CV_AA);
				cv::putText(frame, gstr, Point(15, 15), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 0), 1, CV_AA);
			}
			frame = fitRedLine(m_data_pts, frame);
		}
    }
    cv::imshow("PolygonDemo", frame);
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

bool PolygonDemo::fitGreenLine(const std::vector<cv::Point>& pts, cv::Vec4i& green, std::string& gstr) 
{
	int n = (int)pts.size();
	if (n < 2) return false;

	// initialize
	cv::Mat GA = cv::Mat(n, 2, CV_64FC1);
	cv::Mat GB = cv::Mat(n, 1, CV_64FC1);

	for (int i = 0; i < n; i++){
		GA.at<double>(i, 0) = pts[i].x;
		GA.at<double>(i, 1) = 1;
		GB.at<double>(i, 0) = pts[i].y;
	}
	
	// Ax = b for green(x = A^+b), y=ax+b
	cv::Mat Gx, GAp;
	double Ga, Gb;
	cv::Point2i Gx_inter, Gy_inter;

	// LS
	GAp = (GA.t() * GA).inv() * GA.t();
	Gx = GAp * GB;
	Ga = Gx.at<double>(0, 0);
	Gb = Gx.at<double>(1, 0);
	
	// green line intercepts
	Gx_inter = cv::Point2i(-Gb / Ga, 0);
	Gy_inter = cv::Point2i(0, Gb);
	if (Gx_inter.x < 0)
		Gx_inter = cv::Point2i(640, 640 * Ga + Gb);
	if (Gy_inter.y < 0)
		Gy_inter = cv::Point2i((480 - Gb) / Ga, 480);
	if (Ga != Ga){ // IND
		double mean = 0;
		for (int i = 0; i < n; i++)
			mean += pts[i].x;
		mean /= (double)pts.size();
		green = cv::Vec4i(mean, 0, mean, 480);
		gstr = "x=" + to_string(mean);
	}
	else {
		green = cv::Vec4i(Gx_inter.x, Gx_inter.y, Gy_inter.x, Gy_inter.y);
		gstr = "y=" + to_string(Ga) + "x+" + to_string(Gb);
	}
}

cv::Mat PolygonDemo::fitRedLine(const std::vector<cv::Point>& pts, cv::Mat &frame)
{
	int n = (int)pts.size();
	if (n < 2) return frame;

	cv::Mat RA = cv::Mat(n, 2, CV_64FC1);
	cv::Mat RY = cv::Mat(n, 1, CV_64FC1);

	for (int i = 0; i < n; i++){
		RA.at<double>(i, 0) = pts[i].x;
		RA.at<double>(i, 1) = 1;
		RY.at<double>(i, 0) = pts[i].y;
	}

	// Ax = b for red(x = A^+b), y=ax+b, Robust Weighted LS
	Mat RP, RApinv;
	double Ra, Rb, tRa = 0, tRb = 0;
	Point2i Rx_inter, Ry_inter;
	string rstr;
	Vec4i red;

	// initial LS
	RApinv = (RA.t() * RA).inv() * RA.t();
	RP = RApinv * RY;

	double *r = new double[n];
	Mat w = Mat::zeros(n, n, CV_64FC1);
	Mat temp, frame_clone;

	// repeat 100 times
	for (int i = 0; i < 200; i++){
		frame_clone = frame.clone();
		for (int j = 0; j < n; j++){
			temp = RA.row(j) * RP;
			r[j] = pts[j].y - temp.at<double>(0, 0);
			w.at<double>(j, j) = 1 / (abs(r[j] / 1.3998) + 1);
		}
		RP = (RA.t() * w * RA).inv() * RA.t() * w * RY;
		Ra = RP.at<double>(0, 0);
		Rb = RP.at<double>(1, 0);
		// green line intercepts
		Rx_inter = cv::Point2i(-Rb / Ra, 0);
		Ry_inter = cv::Point2i(0, Rb);

		if (Rx_inter.x < 0)
			Rx_inter = cv::Point2i(640, 640 * Ra + Rb);
		if (Ry_inter.y < 0)
			Ry_inter = cv::Point2i((480 - Rb) / Ra, 480);

		if (Ra != Ra){ // IND
			double mean = 0;
			for (int i = 0; i < n; i++)
				mean += pts[i].x;
			mean /= (double)pts.size();
			red = cv::Vec4i(mean, 0, mean, 480);
			rstr = "x=" + to_string(mean);
		}
		else {
			red = cv::Vec4i(Rx_inter.x, Rx_inter.y, Ry_inter.x, Ry_inter.y);
			rstr = "y=" + to_string(Ra) + "x+" + to_string(Rb);
		}
		cv::line(frame_clone, Point(red[0], red[1]), Point(red[2], red[3]), Scalar(0, 0, 255), 1, CV_AA);
		cv::putText(frame_clone, rstr, Point(15, 45), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255), 1, CV_AA);
		cv::imshow("PolygonDemo", frame_clone);
		waitKey(100);
		if (Ra - tRa <= 0.00001 && Rb - tRb <= 0.00001) {
			cout << "break : " + to_string(i) + "th" << endl;
			break;
		}
		tRa = Ra; tRb = Rb;
	}
	delete r;
	return frame_clone;
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
	center = cv::Point2d(a, b);

	std::cout << A << std::endl;
	std::cout << C << std::endl;
	std::cout << pinvA << std::endl;
	std::cout << B << std::endl;
	
    return true;
}

bool PolygonDemo::fitEllipse(const std::vector<cv::Point>& pts, cv::Point2d& center, cv::Size &radius, double &angle)
{
	int n = (int)pts.size();
	if (n < 6) return false;

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

	center = cv::Point2d(x0, y0);
	radius = cv::Size(w, h);
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
		m_data_pts.clear();
		m_test_pts.clear();
		m_data_ready = false;
		refreshWindow();
		if (wheel_flag % 4 == 0){      // y=n
			m_data_pts.push_back(Point(100, 100));
			m_data_pts.push_back(Point(200, 100));
			m_data_pts.push_back(Point(300, 100));
			m_data_pts.push_back(Point(400, 100));
			m_data_pts.push_back(Point(500, 100));
		}
		if (wheel_flag % 4 == 1){      // x=n
			m_data_pts.push_back(Point(300, 100));
			m_data_pts.push_back(Point(300, 150));
			m_data_pts.push_back(Point(300, 200));
			m_data_pts.push_back(Point(300, 250));
			m_data_pts.push_back(Point(300, 300));
		}
		if (wheel_flag % 4 == 2){      // normal case
			m_data_pts.push_back(Point(100, 100));
			m_data_pts.push_back(Point(200, 100));
			m_data_pts.push_back(Point(300, 100));
			m_data_pts.push_back(Point(400, 100));
			m_data_pts.push_back(Point(500, 100));
			m_data_pts.push_back(Point(100, 300));
			m_data_pts.push_back(Point(200, 300));
			m_data_pts.push_back(Point(300, 300));
			m_data_pts.push_back(Point(400, 300));
			m_data_pts.push_back(Point(500, 300));
		}
		if (wheel_flag % 4 == 3){      // normal case
			m_data_pts.push_back(Point(300, 50));
			m_data_pts.push_back(Point(300, 150));
			m_data_pts.push_back(Point(300, 250));
			m_data_pts.push_back(Point(300, 350));
			m_data_pts.push_back(Point(500, 50));
			m_data_pts.push_back(Point(500, 150));
			m_data_pts.push_back(Point(500, 250));
			m_data_pts.push_back(Point(500, 350));
		}
		wheel_flag++;
		m_data_ready = true;
		refreshWindow();
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
