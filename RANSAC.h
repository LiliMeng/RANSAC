//Author Lili Meng
//All rights reserved


#include <stdio.h>
#include "cvxImage_310.hpp"
//#include "SCRF_util.hpp"
#include <vector>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <fstream>
#include <iomanip>
#include <sstream>

using std::vector;

using namespace std;


class readData{
public:
    readData(string filename);
    vector<vector<double> > allDataPointsVec;
    vector<cv::Point2d> img_pts;
    vector<cv::Point3d> pred_wld_pts;
    vector<cv::Point3d> gt_wld_pts;
    vector<cv::Point3d> pred_color;
    vector<cv::Point3d> actual_color;
};

class readPose{
public:
    readPose();
    Mat getPose(const char *file_name);
};
