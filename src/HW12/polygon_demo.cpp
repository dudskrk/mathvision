#include "polygon_demo.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <random>
#include <fstream>
#include <sstream>

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
			frame = fitLine(m_data_pts, frame, 20, true);
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

cv::Mat PolygonDemo::fitLine(const std::vector<cv::Point>& pts, cv::Mat &frame, int T, bool optional)
{
	int n = (int)pts.size(), c_max = 0, c_sum = 0;
	if (n < 2) return frame;

	vector<Point> pts_inlier, pts_sampled;
	random_device rd;
	mt19937 gen(rd());
	uniform_int_distribution<int> dis(0, n - 1);
	Mat tA, tP, tw, tu, tvt;
	Mat RP, Rw, Ru, Rvt;
	Mat OP, Ow, Ou, Ovt;
	Mat frame_clone;
	double ta, tb, tc, Ra, Rb, Rc, Oa, Ob, Oc;
	int rd1, rd2, temp;
	Point2i tx_inter, ty_inter, Rx_inter, Ry_inter, Ox_inter, Oy_inter;
	string str, rstr;
	Vec4i sampled_line, ransac;
	
	for (int i = 0; i < 200; i++){
		printf("%d out of %d\r", i + 1, 200);
		frame_clone = frame.clone();
		pts_sampled.clear();
		c_sum = 0;

		do {
			rd1 = dis(gen);
			rd2 = dis(gen);
		} while (rd1 == rd2);
		pts_sampled.push_back(pts[rd1]);
		pts_sampled.push_back(pts[rd2]);
		for (int i = 0; i < 2; i++) {
			circle(frame, pts_sampled[i], 3, Scalar(127, 127, 0), CV_FILLED);
			//cout << pts_sampled[i] << endl;
		}
		tA = Mat(2, 3, CV_64FC1);
		double ta, tb, tc;
		for (int i = 0; i < 2; i++){
			tA.at<double>(i, 0) = pts_sampled[i].x;
			tA.at<double>(i, 1) = pts_sampled[i].y;
			tA.at<double>(i, 2) = 1;
		}

		SVD::compute(tA, tw, tu, tvt, SVD::FULL_UV);
		tP = tvt.row(tvt.rows - 1);
		ta = tP.at<double>(0, 0);
		tb = tP.at<double>(0, 1);
		tc = tP.at<double>(0, 2);
		tx_inter = cv::Point2i(-tc / ta, 0);
		ty_inter = cv::Point2i(0, -tc / tb);
		if (tx_inter.x < 0)
			tx_inter = cv::Point2i(640, (ta * 640 + tc) / -tb);
		if (ty_inter.y < 0)
			ty_inter = cv::Point2i((480 * tb + tc) / -ta, 480);

		for (int i = 0; i < n; i++){
			if (abs(-ta / tb * pts[i].x - tc / tb - pts[i].y) <= T) {
				c_sum++;
			}
		}
		if (c_max < c_sum) {
			pts_inlier.clear();
			c_max = c_sum;
			Ra = ta; Rb = tb; Rc = tc;
			for (int i = 0; i < n; i++) {
				if (abs(-ta / tb * pts[i].x - tc / tb - pts[i].y) <= T)
					pts_inlier.push_back(pts[i]);
			}
		}
		sampled_line = Vec4i(tx_inter.x, tx_inter.y, ty_inter.x, ty_inter.y);
		rstr = to_string(ta) + "x+" + to_string(tb) + "y+" + to_string(tc) + "=0";
		line(frame_clone, Point(sampled_line[0], sampled_line[1]), Point(sampled_line[2], sampled_line[3]), Scalar(0, 255, 0), 1, CV_AA);
		putText(frame_clone, rstr, Point(15, 15), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 0), 1, CV_AA);
		imshow("PolygonDemo", frame_clone);
		waitKey(20);
	}
	frame_clone = frame.clone();
	if (optional) {
		int m = pts_inlier.size();
		cout << endl << n << ", " << m << endl;
		cv::Mat OA = cv::Mat(m, 3, CV_64FC1);

		for (int i = 0; i < m; i++){
			circle(frame_clone, pts_inlier[i], 3, Scalar(0, 0, 255), CV_FILLED);
			OA.at<double>(i, 0) = pts_inlier[i].x;
			OA.at<double>(i, 1) = pts_inlier[i].y;
			OA.at<double>(i, 2) = 1;
		}

		SVD::compute(OA, Ow, Ou, Ovt, SVD::FULL_UV);
		// LS
		OP = Ovt.row(Ovt.rows - 1);
		Oa = OP.at<double>(0, 0);
		Ob = OP.at<double>(0, 1);
		Oc = OP.at<double>(0, 2);

		// Ransac line intercepts
		Ox_inter = cv::Point2i(-Oc / Oa, 0);
		Oy_inter = cv::Point2i(0, -Oc / Ob);

		if (Ox_inter.x < 0)
			Ox_inter = cv::Point2i(640, (Oa * 640 + Oc) / -Ob);
		if (Oy_inter.y < 0)
			Oy_inter = cv::Point2i((480 * Ob + Oc) / -Oa, 480);

		ransac = cv::Vec4i(Ox_inter.x, Ox_inter.y, Oy_inter.x, Oy_inter.y);
		str = to_string(Oa) + "x+" + to_string(Ob) + "y+" + to_string(Oc) + "=0";
		line(frame_clone, Point(ransac[0], ransac[1]), Point(ransac[2], ransac[3]), Scalar(0, 255, 0), 1, CV_AA);
		putText(frame_clone, str, Point(15, 45), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 0), 1, CV_AA);
		imshow("PolygonDemo", frame_clone);
	}
	// Ransac line intercepts
	Rx_inter = cv::Point2i(-Rc / Ra, 0);
	Ry_inter = cv::Point2i(0, -Rc / Rb);

	if (Rx_inter.x < 0)
		Rx_inter = cv::Point2i(640, (Ra * 640 + Rc) / -Rb);
	if (Ry_inter.y < 0)
		Ry_inter = cv::Point2i((480 * Rb + Rc) / -Ra, 480);

	ransac = cv::Vec4i(Rx_inter.x, Rx_inter.y, Ry_inter.x, Ry_inter.y);
	str = to_string(Ra) + "x+" + to_string(Rb) + "y+" + to_string(Rc) + "=0";
	line(frame_clone, Point(ransac[0], ransac[1]), Point(ransac[2], ransac[3]), Scalar(0, 0, 255), 1, CV_AA);
	putText(frame_clone, str, Point(15, 15), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255), 1, CV_AA);
	imshow("PolygonDemo", frame_clone);
	waitKey(0);
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
		//std::printf("%d %d\n", x, y);
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
		wheel_flag = 0;
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
		/* // for same data, itertaion, N, T
		m_data_ready = false;
		if (wheel_flag == 0){      
			m_copy_pts.resize(m_data_pts.size());
			copy(m_data_pts.begin(), m_data_pts.end(), m_copy_pts.begin());
		}
		if (wheel_flag >= 1){      
			m_data_pts.resize(m_copy_pts.size());
			copy(m_copy_pts.begin(), m_copy_pts.end(), m_data_pts.begin());
		}
		wheel_flag++;
		m_data_ready = true;
		refreshWindow();
		*/
		// for same data
		m_data_ready = false;
		if (wheel_flag == 0){
			ofstream out("point50.txt");
			if (out.is_open()){
				for (int i = 0; i < m_data_pts.size(); i++){
					out << m_data_pts[i].x << " " << m_data_pts[i].y << endl;
				}
			}
			out.close();
			m_data_pts.clear();
		}
		if (wheel_flag >= 1){
			m_data_pts.clear();
			ifstream in("points50.txt");
			string str;
			stringstream ss;
			int x, y;
			if (in.is_open()){
				while (in){
					getline(in, str);
					ss.str(str);
					ss >> x >> y;
					ss.clear();
					m_data_pts.push_back(Point(x, y));
				}
			}
			in.close();
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
