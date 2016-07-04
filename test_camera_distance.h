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
#include "SCRF_tree.hpp"
#include "cvx_io.hpp"
#include <iostream>
#include <fstream>
#include "SCRF_regressor_builder.hpp"
#include "SCRF_regressor.hpp"
#include "RGBG_tree.hpp"
#include "RGBGUtil.hpp"
#include "RGBGRegressor.hpp"
#include "RGBGRegressorBuilder.hpp"
#include "cvxPoseEstimation.hpp"
#include "ms7ScenesUtil.hpp"

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

void test_camera_distance();


#endif /* defined(__RGBRegressionForest__test_camera_distance__) */
