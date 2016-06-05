//
//  cvx_io.cpp
//  LoopClosure
//
//  Created by jimmy on 2016-03-20.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "cvx_io.hpp"
#include <iostream>
#include <dirent.h>

using cv::Mat;
using std::cout;
using std::endl;


bool cvx_io::imread_depth_16bit_to_32f(const char *file, cv::Mat & depth_img)
{
    depth_img = cv::imread(file, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    if (depth_img.empty()) {
        printf("Error: can not read image from %s\n", file);
        return false;
    }
    assert(depth_img.type() == CV_16UC1);
    depth_img.convertTo(depth_img, CV_32F);
    return true;
}

bool cvx_io::imread_depth_16bit_to_64f(const char *filename, cv::Mat & depth_img)
{
    depth_img = cv::imread(filename, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    if (depth_img.empty()) {
        printf("Error: can not read image from %s\n", filename);
        return false;
    }
    assert(depth_img.type() == CV_16UC1);
    depth_img.convertTo(depth_img, CV_64F);
    return true;
}

bool cvx_io::imread_rgb_8u(const char *file_name, cv::Mat & rgb_img)
{
    rgb_img = cv::imread(file_name, CV_LOAD_IMAGE_COLOR);
    if (rgb_img.empty()) {
        printf("Error: can not read image from %s\n", file_name);
        return false;
    }
    assert(rgb_img.type() == CV_8UC3);
    return true;
}

void cvx_io::imwrite_depth_8u(const char *file, const cv::Mat & depth_img)
{
    assert(depth_img.type() == CV_32F || depth_img.type() == CV_64F);
    assert(depth_img.channels() == 1);

    double minv = 0.0;
    double maxv = 0.0;
    cv::minMaxLoc(depth_img, &minv, &maxv);

    printf("min, max values are: %lf %lf\n", minv, maxv);


    cv::Mat shifted_depth_map;
    depth_img.convertTo(shifted_depth_map, CV_32F, 1.0, -minv);
    cv::Mat depth_8u;
    shifted_depth_map.convertTo(depth_8u, CV_8UC1, 255/(maxv - minv));

    cv::imwrite(file, depth_8u);
    printf("save to: %s\n", file);
}

bool cvx_io::save_mat(const char *txtfile, const cv::Mat & mat)
{
    assert(mat.type() == CV_64FC1);
    FILE * pf = fopen(txtfile, "w");
    if (!pf) {
        printf("Error: can not write to %s \n", txtfile);
        return false;
    }
    fprintf(pf, "%d %d\n", mat.rows, mat.cols);
    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x< mat.cols; x++) {
            fprintf(pf, "%lf ", mat.at<double>(y, x));
        }
        fprintf(pf, "\n");
    }
    fclose(pf);
    printf("save to %s\n", txtfile);
    return true;
}

bool cvx_io::load_mat(const char *txtfile, cv::Mat & mat)
{
    return true;
}

vector<string> cvx_io::read_files(const char *dir_name)
{

    const char *post_fix = strrchr(dir_name, '.');
    string pre_str(dir_name);
    pre_str = pre_str.substr(0, pre_str.rfind('/') + 1);
    //printf("pre_str is %s\n", pre_str.c_str());

    assert(post_fix);
    vector<string> file_names;
    DIR *dir = NULL;
    struct dirent *ent = NULL;
    if ((dir = opendir (pre_str.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            const char *cur_post_fix = strrchr( ent->d_name, '.');
            //printf("cur post_fix is %s %s\n", post_fix, cur_post_fix);

            if (!strcmp(post_fix, cur_post_fix)) {
                file_names.push_back(pre_str + string(ent->d_name));
                cout<<file_names.back()<<endl;
            }

            //printf ("%s\n", ent->d_name);
        }
        closedir (dir);
    }
    printf("read %lu files\n", file_names.size());
    return file_names;
}



/********     ms_7_scenes_util      ************/
Mat ms_7_scenes_util::read_pose_7_scenes(const char *file_name)
{
    Mat P = Mat::zeros(4, 4, CV_64F);
    FILE *pf = fopen(file_name, "r");
    assert(pf);
    for (int row = 0; row<4; row++) {
        for (int col = 0; col<4; col++) {
            double v = 0;
            fscanf(pf, "%lf", &v);
            P.at<double>(row, col) = v;
        }
    }
    fclose(pf);
//    cout<<"pose is "<<P<<endl;
    return P;
}


// return CV_64F
Mat ms_7_scenes_util::camera_depth_to_world_depth(const cv::Mat & camera_depth_img, const cv::Mat & pose)
{
    const int width  = camera_depth_img.cols;
    const int height = camera_depth_img.rows;
    Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 585.0;
    K.at<double>(1, 1) = 585.0;
    K.at<double>(0, 2) = 320.0;
    K.at<double>(1, 2) = 240.0;

    Mat inv_K = K.inv();

    cv::Mat world_depth_img = cv::Mat::zeros(height, width, CV_64F);
    Mat loc_img = cv::Mat::zeros(3, 1, CV_64F);
    Mat loc_camera_h = cv::Mat::zeros(4, 1, CV_64F); // homography coordinate
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            double camera_depth = camera_depth_img.at<double>(r, c)/1000.0;
            if ((int)camera_depth == 65535) {
                // invalid depth
                continue;
            }
            loc_img.at<double>(0, 0) = c;
            loc_img.at<double>(1, 0) = r;
            loc_img.at<double>(2, 0) = 1.0;
            Mat loc_camera = inv_K * loc_img;
            double z = loc_camera.at<double>(2, 0);

            double scale = camera_depth/z;

            loc_camera_h.at<double>(0, 0) = loc_camera.at<double>(0, 0) * scale;
            loc_camera_h.at<double>(1, 0) = loc_camera.at<double>(1, 0) * scale;
            loc_camera_h.at<double>(2, 0) = loc_camera.at<double>(2, 0) * scale;
            loc_camera_h.at<double>(3, 0) = 1.0;

            Mat x_world = pose * loc_camera_h;
            x_world /= x_world.at<double>(3, 0);
            world_depth_img.at<double>(r, c) = x_world.at<double>(2, 0); // save depth in world coordinate
        }
    }
    return world_depth_img;
}

cv::Mat ms_7_scenes_util::camera_depth_to_world_coordinate(const cv::Mat & camera_depth_img, const cv::Mat & camera_to_world_pose)
{
    const int width  = camera_depth_img.cols;
    const int height = camera_depth_img.rows;
    Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 585.0;
    K.at<double>(1, 1) = 585.0;
    K.at<double>(0, 2) = 320.0;
    K.at<double>(1, 2) = 240.0;

    Mat inv_K = K.inv();

    cv::Mat world_coordinate_img = cv::Mat::zeros(height, width, CV_64FC3);
    Mat loc_img = cv::Mat::zeros(3, 1, CV_64F);
    Mat loc_camera_h = cv::Mat::zeros(4, 1, CV_64F); // homography coordinate
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            double camera_depth = camera_depth_img.at<double>(r, c)/1000.0; // to meter
            if ((int)camera_depth == 65535 || camera_depth < 0.001) {
                // invalid depth
                //printf("invalid depth %lf\n", camera_depth);
                continue;
            }
            loc_img.at<double>(0, 0) = c;
            loc_img.at<double>(1, 0) = r;
            loc_img.at<double>(2, 0) = 1.0;
            Mat loc_camera = inv_K * loc_img;
            double z = loc_camera.at<double>(2, 0);
            double scale = camera_depth/z;
            loc_camera_h.at<double>(0, 0) = loc_camera.at<double>(0, 0) * scale;
            loc_camera_h.at<double>(1, 0) = loc_camera.at<double>(1, 0) * scale;
            loc_camera_h.at<double>(2, 0) = loc_camera.at<double>(2, 0) * scale;
            loc_camera_h.at<double>(3, 0) = 1.0;

            Mat x_world = camera_to_world_pose * loc_camera_h;
            x_world /= x_world.at<double>(3, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[0] = x_world.at<double>(0, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[1] = x_world.at<double>(1, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[2] = x_world.at<double>(2, 0);
        }
    }
    //world_coordinate_img /= 1000.0;
    return world_coordinate_img;
}

cv::Mat ms_7_scenes_util::camera_depth_to_world_coordinate(const cv::Mat & camera_depth_img,
                                                           const cv::Mat & camera_to_world_pose,
                                                           cv::Mat & mask)
{
    assert(camera_depth_img.type() == CV_64FC1);
    const int width  = camera_depth_img.cols;
    const int height = camera_depth_img.rows;
    Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 585.0;
    K.at<double>(1, 1) = 585.0;
    K.at<double>(0, 2) = 320.0;
    K.at<double>(1, 2) = 240.0;

    Mat inv_K = K.inv();

    //cout<<"invet K is "<<inv_K<<endl;

    cv::Mat world_coordinate_img = cv::Mat::zeros(height, width, CV_64FC3);
    Mat loc_img = cv::Mat::zeros(3, 1, CV_64F);
    Mat loc_camera_h = cv::Mat::zeros(4, 1, CV_64F); // homography coordinate
    mask = cv::Mat::ones(height, width, CV_8UC1);
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            double camera_depth = camera_depth_img.at<double>(r, c)/1000.0; // to meter
            if (camera_depth == 65.535 || camera_depth < 0.1 || camera_depth > 10.0) {
                // invalid depth
                //printf("invalid depth %lf\n", camera_depth);
                mask.at<unsigned char>(r, c) = 0;
                continue;
            }
            loc_img.at<double>(0, 0) = c;
            loc_img.at<double>(1, 0) = r;
            loc_img.at<double>(2, 0) = 1.0;
            Mat loc_camera = inv_K * loc_img;
            double local_z = loc_camera.at<double>(2, 0);
            double scale = camera_depth/local_z;
            //cout<<"scale is "<<scale<<endl;
            loc_camera_h.at<double>(0, 0) = loc_camera.at<double>(0, 0) * scale;
            loc_camera_h.at<double>(1, 0) = loc_camera.at<double>(1, 0) * scale;
            loc_camera_h.at<double>(2, 0) = loc_camera.at<double>(2, 0) * scale;
            loc_camera_h.at<double>(3, 0) = 1.0;

            Mat x_world = camera_to_world_pose * loc_camera_h;
            x_world /= x_world.at<double>(3, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[0] = x_world.at<double>(0, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[1] = x_world.at<double>(1, 0);
            world_coordinate_img.at<cv::Vec3d>(r, c)[2] = x_world.at<double>(2, 0);
        }
    }

    return world_coordinate_img;
}


bool ms_7_scenes_util::load_prediction_result(const char *file_name, string & rgb_img_file, string & depth_img_file, string & camera_pose_file,
                                              vector<cv::Point2d> & img_pts,
                                              vector<cv::Point3d> & wld_pts_pred,
                                              vector<cv::Point3d> & wld_pts_gt)
{
    assert(file_name);
    FILE *pf = fopen(file_name, "r");
    if (!pf) {
        printf("Error, can not read from %s\n", file_name);
        return false;
    }

    {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        rgb_img_file = string(buf);
    }

    {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        depth_img_file = string(buf);
    }

    {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        camera_pose_file = string(buf);
    }

    {
        char dummy_buf[1024] = {NULL};
        fgets(dummy_buf, sizeof(dummy_buf), pf);
        printf("%s\n", dummy_buf);
    }

    while (1) {
        double val[8] = {0.0};
        int ret = fscanf(pf, "%lf %lf %lf %lf %lf %lf %lf %lf", &val[0], &val[1],
                         &val[2], &val[3], &val[4],
                         &val[5], &val[6], &val[7]);
        if (ret != 8) {
            break;
        }
        // filter out zero points
        img_pts.push_back(cv::Point2f(val[0], val[1]));
        wld_pts_pred.push_back(cv::Point3f(val[2], val[3], val[4]));
        wld_pts_gt.push_back(cv::Point3f(val[5], val[6], val[7]));
    }
    fclose(pf);
    printf("read %lu prediction and ground truth points.\n", wld_pts_gt.size());

    return true;
}


bool ms_7_scenes_util::load_prediction_result_with_color(const char *file_name,
                                                  string & rgb_img_file,
                                                  string & depth_img_file,
                                                  string & camera_pose_file,
                                                  vector<cv::Point2d> & img_pts,
                                                  vector<cv::Point3d> & wld_pts_pred,
                                                  vector<cv::Point3d> & wld_pts_gt,
                                                  vector<cv::Vec3d> & color_pred,
                                                  vector<cv::Vec3d> & color_sample)

{
    assert(file_name);
    FILE *pf = fopen(file_name, "r");
    if (!pf) {
        printf("Error, can not read from %s\n", file_name);
        return false;
    }

    {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        rgb_img_file = string(buf);
    }

    {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        depth_img_file = string(buf);
    }

    {
        char buf[1024] = {NULL};
        fscanf(pf, "%s\n", buf);   // remove the last \n
        camera_pose_file = string(buf);
    }

    {
        char dummy_buf[1024] = {NULL};
        fgets(dummy_buf, sizeof(dummy_buf), pf);
        printf("%s\n", dummy_buf);
    }

    while (1) {
        double val[8] = {0.0};
        int ret = fscanf(pf, "%lf %lf %lf %lf %lf %lf %lf %lf", &val[0], &val[1],
                         &val[2], &val[3], &val[4],
                         &val[5], &val[6], &val[7]);
        if (ret != 8) {
            break;
        }

        // 2D , 3D position
        img_pts.push_back(cv::Point2d(val[0], val[1]));
        wld_pts_pred.push_back(cv::Point3d(val[2], val[3], val[4]));
        wld_pts_gt.push_back(cv::Point3d(val[5], val[6], val[7]));

        double val2[6] = {0.0};
        ret = fscanf(pf, "%lf %lf %lf %lf %lf %lf",
                     &val2[0], &val2[1], &val2[2],
                     &val2[3], &val2[4], &val2[5]);
        if (ret != 6) {
            break;
        }
        color_pred.push_back(cv::Vec3d(val2[0], val2[1], val2[2]));
        color_sample.push_back(cv::Vec3d(val2[3], val2[4], val2[5]));
        assert(img_pts.size() == color_pred.size());
    }
    fclose(pf);
    printf("read %lu prediction and ground truth points.\n", wld_pts_gt.size());

    return true;


}


