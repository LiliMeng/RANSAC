//
//  cvx_io.hpp
//  LoopClosure
//
//  Created by jimmy on 2016-03-20.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef cvx_io_cpp
#define cvx_io_cpp

// open CV input and output
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include <string>
#include <vector>

using std::string;
using std::vector;

class cvx_io
{
public:
    // unit: millimeter
    static bool imread_depth_16bit_to_32f(const char *file, cv::Mat & depth_img);
    // unit: millimeter
    static bool imread_depth_16bit_to_64f(const char *filename, cv::Mat & depth_img);
    static bool imread_rgb_8u(const char *file_name, cv::Mat & rgb_img);

    // write depth image as 8u for visualization purpose
    static void imwrite_depth_8u(const char *file, const cv::Mat & depth_img);

    static bool save_mat(const char *txtfile, const cv::Mat & mat);
    static bool load_mat(const char *txtfile, cv::Mat & mat);

    // dir_name = "/Users/jimmy/*.txt"
    static vector<string> read_files(const char *dir_name);
};

class CvxUtil
{
public:
    static inline bool isInside(const int width, const int height, const int x, const int y)
    {
        return x >= 0 && y >=0 && x< width && y < height;
    }
};


// all Mat type is 64F
class ms_7_scenes_util
{
public:
    // read camera pose file
    static cv::Mat read_pose_7_scenes(const char *file_name);

    // invalid depth is 0.0
    static cv::Mat camera_depth_to_world_depth(const cv::Mat & camera_depth_img, const cv::Mat & pose);

    // camera_depth_img 16 bit
    // return CV_64_FC3 for x, y, z, unit in meter
    static cv::Mat camera_depth_to_world_coordinate(const cv::Mat & camera_depth_img, const cv::Mat & camera_to_world_pose);

    // mask: CV_8UC1 0 --> invalid sample
    static cv::Mat camera_depth_to_world_coordinate(const cv::Mat & camera_depth_img,
                                                    const cv::Mat & camera_to_world_pose,
                                                    cv::Mat & mask);


    static inline int invalid_camera_depth(){return 65535;}

    static bool load_prediction_result(const char *file_name, string & rgb_img_file, string & depth_img_file, string & camera_pose_file,
                                       vector<cv::Point2d> & img_pts,
                                       vector<cv::Point3d> & wld_pts_pred,
                                       vector<cv::Point3d> & wld_pts_gt);

    // load prediction result from decision trees with color information
    static bool load_prediction_result_with_color(const char *file_name,
                                                  string & rgb_img_file,
                                                  string & depth_img_file,
                                                  string & camera_pose_file,
                                                  vector<cv::Point2d> & img_pts,
                                                  vector<cv::Point3d> & wld_pts_pred,
                                                  vector<cv::Point3d> & wld_pts_gt,
                                                  vector<cv::Vec3d> & color_pred,
                                                  vector<cv::Vec3d> & color_sample);
};




#endif /* cvx_io_cpp */

