#include "polygon_demo.hpp"
#include "opencv2/imgproc.hpp"
#include "iostream"

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
			case TWIST_REFLECTION:
				sprintf_s(type_str, 100, "twist reflection");
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
    }

    imshow("PolygonDemo", frame);
}

// return the area of polygon
int PolygonDemo::polyArea(const std::vector<cv::Point>& vtx)
{
	int area = 0;
	Point O = vtx[0];
	for (int i = 1; i < vtx.size() - 1; i++){
		Point v1 = vtx[i];
		Point v2 = vtx[i + 1];
		area += ((v1.x - O.x) * (v2.y - v1.y) - (v2.x - v1.x) * (v1.y - O.y)) / 2;
	}

    return abs(area);
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
	int J_p[4], J_q[4];
	for (int i = 0; i < 4; i++){
		J_p[i] = ((pts1[i % 4].x - pts1[(i + 1) % 4].x) * (pts1[(i + 2) % 4].y - pts1[(i + 1) % 4].y) 
			- (pts1[i % 4].y - pts1[(i + 1) % 4].y) * (pts1[(i + 2) % 4].x - pts1[(i + 1) % 4].x));
		J_q[i] = ((pts2[i % 4].x - pts2[(i + 1) % 4].x) * (pts2[(i + 2) % 4].y - pts2[(i + 1) % 4].y) 
			- (pts2[i % 4].y - pts2[(i + 1) % 4].y) * (pts2[(i + 2) % 4].x - pts2[(i + 1) % 4].x));
	}

	if (J_p[0] * J_q[0] > 0 && J_p[1] * J_q[1] > 0 && J_p[2] * J_q[2] > 0 && J_p[3] * J_q[3] > 0)
		return NORMAL;
	else if (J_p[0] * J_q[0] < 0 && J_p[1] * J_q[1] > 0 && J_p[2] * J_q[2] > 0 && J_p[3] * J_q[3] > 0)
		return CONCAVE;
	else if (J_p[0] * J_q[0] < 0 && J_p[1] * J_q[1] < 0 && J_p[2] * J_q[2] > 0 && J_p[3] * J_q[3] < 0)
		return CONCAVE_REFLECTION;
	else if (J_p[0] * J_q[0] > 0 && J_p[1] * J_q[1] < 0 && J_p[2] * J_q[2] < 0 && J_p[3] * J_q[3] > 0)
		return TWIST;
	else if (J_p[0] * J_q[0] > 0 && J_p[1] * J_q[1] > 0 && J_p[2] * J_q[2] < 0 && J_p[3] * J_q[3] < 0)
		return TWIST_REFLECTION;
	else if (J_p[0] * J_q[0] < 0 && J_p[1] * J_q[1] > 0 && J_p[2] * J_q[2] > 0 && J_p[3] * J_q[3] < 0)
		return TWIST_REFLECTION;
	else if (J_p[0] * J_q[0] < 0 && J_p[1] * J_q[1] < 0 && J_p[2] * J_q[2] > 0 && J_p[3] * J_q[3] > 0)
		return TWIST_REFLECTION;
	else if (J_p[0] * J_q[0] < 0 && J_p[1] * J_q[1] < 0 && J_p[2] * J_q[2] < 0 && J_p[3] * J_q[3] < 0)
		return REFLECTION;
}

// estimate a circle that best approximates the input points and return center and radius of the estimate circle
bool PolygonDemo::fitCircle(const std::vector<cv::Point>& pts, cv::Point2d& center, double& radius)
{
    int n = (int)pts.size();
    if (n < 3) return false;

    return false;
}

void PolygonDemo::drawPolygon(Mat& frame, const std::vector<cv::Point>& vtx, bool closed)
{
    int i = 0;
    for (i = 0; i < (int)m_data_pts.size(); i++)
    {
        circle(frame, m_data_pts[i], 2, Scalar(255, 255, 255), CV_FILLED);
    }
    for (i = 0; i < (int)m_data_pts.size() - 1; i++)
    {
        line(frame, m_data_pts[i], m_data_pts[i + 1], Scalar(255, 255, 255), 1);
    }
    if (closed)
    {
        line(frame, m_data_pts[i], m_data_pts[0], Scalar(255, 255, 255), 1);
    }
}

void PolygonDemo::handleMouseEvent(int evt, int x, int y, int flags)
{
    if (evt == CV_EVENT_LBUTTONDOWN)
    {
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
		if (wheel_flag % 4 == 0){      // normal case
			m_data_pts.push_back(Point(100, 100));
			m_data_pts.push_back(Point(300, 100));
			m_data_pts.push_back(Point(300, 300));
			m_data_pts.push_back(Point(100, 300));
		}
		else if (wheel_flag % 4 == 1){   // convex
			m_data_pts.push_back(Point(150, 286));
			m_data_pts.push_back(Point(250, 286));
			m_data_pts.push_back(Point(300, 200));
			m_data_pts.push_back(Point(250, 114));
			m_data_pts.push_back(Point(150, 114));
			m_data_pts.push_back(Point(100, 200));
		}
		else if (wheel_flag % 4 == 2){   // concave
			m_data_pts.push_back(Point(150, 286));
			m_data_pts.push_back(Point(150, 236));
			m_data_pts.push_back(Point(250, 236));
			m_data_pts.push_back(Point(250, 286));
			m_data_pts.push_back(Point(300, 200));
			m_data_pts.push_back(Point(250, 114));
			m_data_pts.push_back(Point(150, 114));
			m_data_pts.push_back(Point(100, 200));
		}
		else if (wheel_flag % 4 == 3){   // cross
			m_data_pts.push_back(Point(150, 150));
			m_data_pts.push_back(Point(250, 250));
			//m_data_pts.push_back(Point(350, 150));
			//m_data_pts.push_back(Point(450, 250));
			//m_data_pts.push_back(Point(450, 150));
			//m_data_pts.push_back(Point(350, 250));
			m_data_pts.push_back(Point(250, 150));
			m_data_pts.push_back(Point(150, 250));
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
