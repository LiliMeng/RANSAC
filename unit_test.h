//
//  test_camera_distance.h
//  RGBRegressionForest
//
//  Created by Lili on 7/3/16.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __RGBRegressionForest__test_camera_distance__
#define __RGBRegressionForest__test_camera_distance__


#include "UT_SC_regression.hpp"
#include <iostream>
#include <fstream>
#include "RGBG_tree.hpp"
#include "RGBGUtil.hpp"
#include "RGBGRegressor.hpp"
#include "RGBGRegressorBuilder.hpp"
#include "cvxPoseEstimation.hpp"
#include "ms7ScenesUtil.hpp"
#include "cvxCalib3d.hpp"
#include "cvxIO.hpp"
#include "FeatureMatching.h"


using namespace cv;
using namespace std;


void camera_rot_distance(const char* train_file_name,
                         const char* test_file_name,
                         const double trans_threshold,
                         vector<double> &test_min_rot_vec_within_trans_threshold);

void camera_trans_distance(const char* train_file_name,
                     const char* test_file_name,
                     const double rot_threshold,
                     vector<double> &test_min_trans_vec_within_rot_threshold);

void camera_rot_dis_under_min_trans_dis(const char* train_file_name,
                                        const char* test_file_name,
                                        vector<double> & min_trans_dis_vec,
                                        vector<double> & rot_dis_under_min_trans_dis);

void test_camera_distance();

void test_feature_matching();

void test_TUM_pose();

void test_minCameraDistanceUnderAngularThreshold();

void test_min_camera_rotDistanceUnderTransThreshold();

void test_cameraDepthToWorldCoordinate();

void test_datasetParameter();

void test_rgbgRF_single_tree();

void test_rgbgRF_single_tree_multi_img();

void test_rgbgRF_single_tree_multi_img_train_test_dif_imgs();

void test_rgbg_multi_tree_multiple_images_model();

void test_4_scenes_angular_dis_within_min_trans_dis();




#endif /* defined(__RGBRegressionForest__test_camera_distance__) */
