
//  test_camera_distance.cpp
//  RGBRegressionForest
//
//  Created by Lili on 7/3/16.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "test_camera_distance.h"


struct mycomparision1 {
    bool operator() (double i, double j) {return (i>j);}
}mycompareLS;

struct mycomparision2 {
    bool operator() (double i,double j) { return (i<j);}
} mycompareSL;

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {
    
    //initialize original index locations
    vector<size_t> index(v.size());
    for(size_t i=0; i !=index.size(); ++i)
    {
        index[i]=i;
    }
    
    //sort indexes based on comparing values in v
    sort(index.begin(), index.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; } );
    
    return index;
    
}


void camera_rot_distance(const char* train_file_name,
                         const char* test_file_name,
                         const double trans_threshold,
                         vector<double> &rot_vec_within_trans_threshold)
{
    vector<string> train_pose_files = Ms7ScenesUtil::read_file_names(train_file_name);
    vector<string> test_pose_files  = Ms7ScenesUtil::read_file_names(test_file_name);
    
    
    vector<cv::Mat> train_poses_vec, test_poses_vec;
    for(int i=0; i<test_pose_files.size(); i++)
    {
        const char* test_pose_file = test_pose_files[i].c_str();
        
        cv::Mat test_camera_pose=Ms7ScenesUtil::read_pose_7_scenes(test_pose_file);
        
        test_poses_vec.push_back(test_camera_pose);
        
    }
    
    for(int i=0; i<train_pose_files.size(); i++)
    {
        const char* train_pose_file=train_pose_files[i].c_str();
        
        cv::Mat train_camera_pose=Ms7ScenesUtil::read_pose_7_scenes(train_pose_file);
        
        train_poses_vec.push_back(train_camera_pose);
        
    }
    
    
    for(int k=0; k<test_poses_vec.size(); k++)
    {
        cv::Mat pose1=test_poses_vec[k];
        //cout<<"the pose file from the test sequence is "<<test_pose_files[k]<<endl;
        
        for(int i=0; i<train_poses_vec.size();i++)
        {
            
            double rot_error = 0.0;
            double trans_error = 0.0;
            
            
            CvxPoseEstimation::poseDistance(pose1,
                                            train_poses_vec[i],
                                            rot_error,
                                            trans_error);
            
        
            if (trans_error < trans_threshold) {
                
                rot_vec_within_trans_threshold.push_back(rot_error);
            }
        }
        
        
        std::sort(rot_vec_within_trans_threshold.begin(), rot_vec_within_trans_threshold.end(), mycompareSL);
        
        
        if(rot_vec_within_trans_threshold.size()>=1)
        {
            
            printf("within the trans threshold of %f, the minimum rot error is %lf\n", trans_threshold, rot_vec_within_trans_threshold[0]);
        
        }
        else
        {
            printf("please increase the trans threshold");
        }
    
    }
}

void camera_rot_dis_under_min_trans_dis(const char* train_file_name,
                                        const char* test_file_name,
                                        vector<double> & min_trans_dis_vec,
                                        vector<double> & rot_dis_under_min_trans_dis)
{
    
    vector<string> train_pose_files = Ms7ScenesUtil::read_file_names(train_file_name);
    vector<string> test_pose_files  = Ms7ScenesUtil::read_file_names(test_file_name);
        
        
    vector<cv::Mat> train_poses_vec, test_poses_vec;
    
    for(int i=0; i<train_pose_files.size(); i++)
    {
        const char* train_pose_file=train_pose_files[i].c_str();
        
        cv::Mat train_camera_pose=Ms7ScenesUtil::read_pose_7_scenes(train_pose_file);
        
        train_poses_vec.push_back(train_camera_pose);
        
    }
    
    for(int i=0; i<test_pose_files.size(); i++)
    {
        const char* test_pose_file = test_pose_files[i].c_str();
            
        cv::Mat test_camera_pose=Ms7ScenesUtil::read_pose_7_scenes(test_pose_file);
            
        test_poses_vec.push_back(test_camera_pose);
            
    }
        
  
        
    for(int k=0; k<test_poses_vec.size(); k++)
    {
        cv::Mat pose1=test_poses_vec[k];
        //cout<<"the pose file from the test sequence is "<<test_pose_files[k]<<endl;
        vector<double>  trans_error_vec, rot_error_vec, original_index;
        for(int i=0; i<train_poses_vec.size();i++)
        {
                
            double rot_error = 0.0;
            double trans_error = 0.0;
                
                
            CvxPoseEstimation::poseDistance(pose1,
                                            train_poses_vec[i],
                                            rot_error,
                                            trans_error);
            original_index.push_back(i);
            trans_error_vec.push_back(trans_error);
            rot_error_vec.push_back(rot_error);
                
        }
        
        assert(train_poses_vec.size()==trans_error_vec.size());
        assert(train_poses_vec.size()==rot_error_vec.size());
     
        vector<size_t> sorted_index_vec=sort_indexes(trans_error_vec);
        std::sort(trans_error_vec.begin(), trans_error_vec.end(), mycompareSL);
        
        double min_trans_error=trans_error_vec[0];
       // cout<<"min_trans_error "<<min_trans_error<<endl;
        
       // cout<<"the index of min_trans_error is "<<sorted_index_vec[min_trans_error]<<endl;
        double rot_error_under_min_trans_error=rot_error_vec[sorted_index_vec[min_trans_error]];
            
        rot_dis_under_min_trans_dis.push_back(rot_error_under_min_trans_error);
        min_trans_dis_vec.push_back(min_trans_error);
            
    }

}


void camera_trans_distance(const char* train_file_name,
                           const char* test_file_name,
                           const double rot_threshold,
                           vector<double> &test_min_trans_vec_within_rot_threshold)
{

    vector<string> train_pose_files = Ms7ScenesUtil::read_file_names(train_file_name);
    vector<string> test_pose_files = Ms7ScenesUtil::read_file_names(test_file_name);

    
    vector<cv::Mat> train_poses_vec, test_poses_vec;
    for(int i=0; i<test_pose_files.size(); i++)
    {
        const char* test_pose_file = test_pose_files[i].c_str();
        
        cv::Mat test_camera_pose=Ms7ScenesUtil::read_pose_7_scenes(test_pose_file);
        
        test_poses_vec.push_back(test_camera_pose);
        
    }
    
    for(int i=0; i<train_pose_files.size(); i++)
    {
        const char* train_pose_file=train_pose_files[i].c_str();
        
        cv::Mat train_camera_pose=Ms7ScenesUtil::read_pose_7_scenes(train_pose_file);
        
        train_poses_vec.push_back(train_camera_pose);
       
    }

    vector<double> inlier_rot_test, inlier_trans_test;
    
    //suppose we just pick up the pose1 from the test files as our query pose, and see the range of other poses in the training set compared with pose1
    
    for(int k=0; k<test_poses_vec.size(); k++)
    {
        cv::Mat pose1=test_poses_vec[k];
        //cout<<"the pose file from the test sequence is "<<test_pose_files[k]<<endl;
    
        vector<double> rot_error_vec, trans_error_vec, trans_error_within_rot_threshold;
    
        vector<size_t> trans_error_rot_threshold_indexes;
        for(int i=0; i<train_poses_vec.size();i++)
        {
        
            double rot_error = 0.0;
            double trans_error = 0.0;
            
            
            CvxPoseEstimation::poseDistance(pose1,
                                            train_poses_vec[i],
                                            rot_error,
                                            trans_error);
        
            rot_error_vec.push_back(rot_error);
            trans_error_vec.push_back(trans_error);
            if (rot_error < rot_threshold) {
                trans_error_rot_threshold_indexes.push_back(i);
                trans_error_within_rot_threshold.push_back(trans_error);
            }
        }
    
        
        std::sort(trans_error_within_rot_threshold.begin(), trans_error_within_rot_threshold.end(), mycompareSL);
        
        vector<size_t> sorted_index_vec=sort_indexes(trans_error_within_rot_threshold);
        
        if(trans_error_within_rot_threshold.size()>=1)
        {
    
            //printf("within the rot threshold of %f, the minimum trans error is %lf\n", rot_threshold, trans_error_within_rot_threshold[0]);
            
           // test_min_trans_vec_within_rot_threshold.push_back(trans_error_within_rot_threshold[0]);
            test_min_trans_vec_within_rot_threshold.push_back(trans_error_within_rot_threshold[0]);
            
            if(trans_error_within_rot_threshold.size()==1)
            {
                printf("within the rot threshold of %f, the minimum trans error is %lf\n", rot_threshold, trans_error_within_rot_threshold[sorted_index_vec[0]]);
                cout<<"the pose file is "<<train_pose_files[trans_error_rot_threshold_indexes[sorted_index_vec[0]]]<<endl;
    
            }
            else if(trans_error_within_rot_threshold.size()==2)
            {
            
                printf("within the rot threshold of %f, the minimum trans error is %lf\n", rot_threshold, trans_error_within_rot_threshold[sorted_index_vec[0]]);
                cout<<"the pose file is "<<train_pose_files[trans_error_rot_threshold_indexes[sorted_index_vec[0]]]<<endl;
                printf("within the rot threshold of %f, the second minimum trans error is %lf\n", rot_threshold, trans_error_within_rot_threshold[sorted_index_vec[1]]);
                cout<<"the pose file is "<<train_pose_files[trans_error_rot_threshold_indexes[sorted_index_vec[1]]]<<endl;
            }
            else if(trans_error_within_rot_threshold.size()>=3)
            {
                printf("within the rot threshold of %f, the minimum trans error is %lf\n", rot_threshold, trans_error_within_rot_threshold[sorted_index_vec[0]]);
                cout<<"the pose file is "<<train_pose_files[trans_error_rot_threshold_indexes[sorted_index_vec[0]]]<<endl;
                printf("within the rot threshold of %f, the 2nd minimum trans error is %lf\n", rot_threshold, trans_error_within_rot_threshold[sorted_index_vec[1]]);
                cout<<"the pose file is "<<train_pose_files[trans_error_rot_threshold_indexes[sorted_index_vec[1]]]<<endl;
                printf("within the rot threshold of %f, the 3rd minimum trans error is %lf\n", rot_threshold, trans_error_within_rot_threshold[sorted_index_vec[2]]);
                cout<<"the pose file is "<<train_pose_files[trans_error_rot_threshold_indexes[sorted_index_vec[2]]]<<endl;
            }
       }
    }
}

void test_feature_matching()
{
    const char* dataset_param_filename="/Users/jimmy/Desktop/SCRF_RGBD_train/SCRF_RGBD_Fire/dataset_param_TUM.txt";
    
    char depth_img_file1[] ="/Users/jimmy/Desktop/images/TUM_SLAM/freiburg3_long_office_household/seq-01/frame-000000.depth.png";
    char rgb_img_file1[]   = "/Users/jimmy/Desktop/images/TUM_SLAM/freiburg3_long_office_household/seq-01/frame-000000.color.png";
    
    char depth_img_file2[] ="/Users/jimmy/Desktop/images/TUM_SLAM/freiburg3_long_office_household/seq-01/frame-000120.depth.png";
    char rgb_img_file2[]   = "/Users/jimmy/Desktop/images/TUM_SLAM/freiburg3_long_office_household/seq-01/frame-000120.color.png";
    
    DatasetParameter dataset_param;
    
    dataset_param.readFromFileDataParameter(dataset_param_filename);
    
    cv::Mat rgb_img1, rgb_img2;
    
    CvxIO::imread_rgb_8u(rgb_img_file1, rgb_img1);
    
    CvxIO::imread_rgb_8u(rgb_img_file2, rgb_img2);
    
    cv::Mat world2camera_coord1, world2camera_coord2;
    
    vector<cv::KeyPoint> keypoints_1, keypoints_2;
    vector<cv::DMatch> good_matches;
    FeatureMatching::descriptor_matching(rgb_img1, rgb_img2, keypoints_1, keypoints_2, good_matches);
}


void test_TUM_pose()
{
    const char* dataset_param_filename="/Users/jimmy/Desktop/SCRF_RGBD_train/SCRF_RGBD_Fire/dataset_param_TUM.txt";
    
    char depth_img_file1[] ="/Users/jimmy/Desktop/images/TUM_SLAM/freiburg3_long_office_household/seq-01/frame-000000.depth.png";
    char rgb_img_file1[]   = "/Users/jimmy/Desktop/images/TUM_SLAM/freiburg3_long_office_household/seq-01/frame-000000.color.png";
    char pose_file1 [] = "/Users/jimmy/Desktop/images/TUM_SLAM/freiburg3_long_office_household/seq-01/frame-000000.pose.txt";
    
    
    DatasetParameter dataset_param;
    
    dataset_param.readFromFileDataParameter(dataset_param_filename);
    
    
    cv::Mat rgb_img1, camera_depth_img1;
    
    CvxIO::imread_rgb_8u(rgb_img_file1, rgb_img1);
    CvxIO::imread_depth_16bit_to_64f(depth_img_file1, camera_depth_img1);
    
    
    cv::Mat camera2world_pose = Ms7ScenesUtil::read_pose_7_scenes(pose_file1);
    cv::Mat world2camera_pose = camera2world_pose.inv();
    FeatureMatching::img_points_to_world_and_world2camera(rgb_img1, camera_depth_img1, dataset_param, world2camera_pose);
    
    
}

void test_min_camera_rotDistanceUnderTransThreshold()
{
    ofstream fout1("/Users/jimmy/Desktop/RGBTrainChess/rotDiswithTransError20_fire.txt");
    ofstream fout2("/Users/jimmy/Desktop/RGBTrainChess/rotDiswithTransError30_fire.txt");
    ofstream fout3("/Users/jimmy/Desktop/RGBTrainChess/rotDiswithTransError50_fire.txt");
    
    // read ground truth data for chess
    //const char* train_file_name="/Users/jimmy/Desktop/RGBTrainChess/train_4000_chess/camera_pose_list.txt";
    //const char* test_file_name = "/Users/jimmy/Desktop/RGBTrainChess/test_2000_chess/camera_pose_list.txt";
    
    ///for heads
    //const char* train_file_name="/Users/jimmy/Desktop/RGBTrainHeads/train_1000/camera_pose_list.txt";
    //const char* test_file_name = "/Users/jimmy/Desktop/RGBTrainHeads/test_1000/camera_pose_list.txt";
    
    ///for fire
    const char* train_file_name="/Users/jimmy/Desktop/RGBTrainFire/camera_pose_list.txt";
    const char* test_file_name = "/Users/jimmy/Desktop/RGBTrainFire/test/camera_pose_list.txt";
    
    double trans_threshold1=0.20;
    double trans_threshold2=0.30;
    double trans_threshold3=0.50;
    
    vector<double> rot_vec_within_trans_threshold1, rot_vec_within_trans_threshold2, rot_vec_within_trans_threshold3;
    camera_rot_distance(train_file_name,
                          test_file_name,
                          trans_threshold1,
                          rot_vec_within_trans_threshold1);
    
    camera_rot_distance(train_file_name,
                          test_file_name,
                          trans_threshold2,
                          rot_vec_within_trans_threshold2);
    
    camera_rot_distance(train_file_name,
                          test_file_name,
                          trans_threshold3,
                          rot_vec_within_trans_threshold3);
    
    
    for(int i=0; i<rot_vec_within_trans_threshold1.size();i++)
    {
        
        fout1<<rot_vec_within_trans_threshold1[i]<<endl;
    }
    
    
    for(int i=0; i<rot_vec_within_trans_threshold2.size();i++)
    {
        
        fout2<<rot_vec_within_trans_threshold2[i]<<endl;
    }
    
    
    for(int i=0; i<rot_vec_within_trans_threshold3.size();i++)
    {
        fout3<<rot_vec_within_trans_threshold3[i]<<endl;
    }
    

}


void test_camera_distance()
{
    ofstream fout1("/Users/jimmy/Desktop/RGBTrainChess/test_trans_error5_fire.txt");
    ofstream fout2("/Users/jimmy/Desktop/RGBTrainChess/test_trans_error10_fire.txt");
    ofstream fout3("/Users/jimmy/Desktop/RGBTrainChess/test_trans_error15_fire.txt");
    
    // read ground truth data for chess
    //const char* train_file_name="/Users/jimmy/Desktop/RGBTrainChess/train_4000_chess/camera_pose_list.txt";
    //const char* test_file_name = "/Users/jimmy/Desktop/RGBTrainChess/test_2000_chess/camera_pose_list.txt";
  
    ///for heads
    //const char* train_file_name="/Users/jimmy/Desktop/RGBTrainHeads/train_1000/camera_pose_list.txt";
    //const char* test_file_name = "/Users/jimmy/Desktop/RGBTrainHeads/test_1000/camera_pose_list.txt";
    
    ///for fire
    const char* train_file_name="/Users/jimmy/Desktop/RGBTrainFire/camera_pose_list.txt";
    const char* test_file_name = "/Users/jimmy/Desktop/RGBTrainFire/test/camera_pose_list.txt";
   
    double rot_threshold1=5;
    double rot_threshold2=10;
    double rot_threshold3=15;
    
    vector<double> test_min_trans_vec_within_rot_threshold1, test_min_trans_vec_within_rot_threshold2, test_min_trans_vec_within_rot_threshold3;
    camera_trans_distance(train_file_name,
                          test_file_name,
                          rot_threshold1,
                          test_min_trans_vec_within_rot_threshold1);
    
    camera_trans_distance(train_file_name,
                          test_file_name,
                          rot_threshold2,
                          test_min_trans_vec_within_rot_threshold2);
    
    camera_trans_distance(train_file_name,
                          test_file_name,
                          rot_threshold3,
                          test_min_trans_vec_within_rot_threshold3);
    
    
    for(int i=0; i<test_min_trans_vec_within_rot_threshold1.size();i++)
    {

        fout1<<test_min_trans_vec_within_rot_threshold1[i]<<endl;
    }
    
    
    for(int i=0; i<test_min_trans_vec_within_rot_threshold2.size();i++)
    {
        
        fout2<<test_min_trans_vec_within_rot_threshold2[i]<<endl;
    }
    
    
    for(int i=0; i<test_min_trans_vec_within_rot_threshold3.size();i++)
    {
        fout3<<test_min_trans_vec_within_rot_threshold3[i]<<endl;
    }
    
}


void test_minCameraDistanceUnderAngularThreshold()
{
    ///for chess
    const char* train_file_name="/Users/jimmy/Desktop/RGBTrainChess/train_4000_chess/camera_pose_list.txt";
    const char* test_file_name = "/Users/jimmy/Desktop/RGBTrainChess/test_2000_chess/camera_pose_list.txt";
    
    
    
    double angular_threshold = 0.3;
    vector<string> train_pose_files =Ms7ScenesUtil::read_file_names(train_file_name);
    vector<string> test_pose_files = Ms7ScenesUtil::read_file_names(test_file_name);
    
    vector<cv::Mat> train_poses_vec, test_poses_vec;
    
    for(int i=0; i<train_pose_files.size(); i++)
    {
        const char* train_pose_file=train_pose_files[i].c_str();
        
        cv::Mat train_camera_pose=Ms7ScenesUtil::read_pose_7_scenes(train_pose_file);
        
        train_poses_vec.push_back(train_camera_pose);
        
    }
    
    for(int i=0; i<test_pose_files.size(); i++)
    {
        const char* test_pose_file = test_pose_files[i].c_str();
        
        cv::Mat test_camera_pose=Ms7ScenesUtil::read_pose_7_scenes(test_pose_file);
        
        //test_poses_vec.push_back(test_camera_pose);
        double minCameraDistance=CvxPoseEstimation::minCameraAngleUnderTranslationalThreshold(train_poses_vec, test_camera_pose, angular_threshold);
        printf("Under angular threshold %lf, min camera distance is: %lf\n", angular_threshold, minCameraDistance);
    }
}

void test_cameraDepthToWorldCoordinate()
{
    Mat calibration_matrix = cv::Mat::eye(3, 3, CV_64F);
    calibration_matrix.at<double>(0, 0) = 535.4;
    calibration_matrix.at<double>(1, 1) = 539.2;
    calibration_matrix.at<double>(0, 2) = 320.1;
    calibration_matrix.at<double>(1, 2) = 247.6;
    
    const char* depth_file_name="/Users/jimmy/Desktop/images/TUM_SLAM/freiburg3_long_office_household/seq-01/frame-000000.depth.png";
    const char* pose_file_name="/Users/jimmy/Desktop/images/TUM_SLAM/freiburg3_long_office_household/seq-01/frame-000000.pose.txt";
    
    cv::Mat camera_depth_img;
    CvxIO::imread_depth_16bit_to_64f(depth_file_name, camera_depth_img);
    cv::Mat camera_to_world_pose = Ms7ScenesUtil::read_pose_7_scenes(pose_file_name);
    double depth_factor = 5000;
    double min_depth = 0.05;
    double max_depth = 10.0;
    
    cv::Mat camera_coordinate;
    cv::Mat mask;
    
    cv::Mat world_coordinate_image=CvxCalib3D::cameraDepthToWorldCoordinate(camera_depth_img,
                                                                            camera_to_world_pose,
                                                                            calibration_matrix,
                                                                            depth_factor,
                                                                            min_depth,
                                                                            max_depth,
                                                                            camera_coordinate,
                                                                            mask);
    
  
    vector<Mat> world_coordinate_img_split;
    cv::split(world_coordinate_image,world_coordinate_img_split);
    cv::Mat world_coordinate_image_x=world_coordinate_img_split[0];
    cv::Mat world_coordinate_image_y=world_coordinate_img_split[1];
    cv::Mat world_coordinate_image_z=world_coordinate_img_split[2];
   
    
    
    const char* world_coordinate_imge_file_x="/Users/jimmy/Desktop/SCRF_RGBD_train/TUM/world_coordinate_imge_file_x.txt";
    const char* world_coordinate_imge_file_y="/Users/jimmy/Desktop/SCRF_RGBD_train/TUM/world_coordinate_imge_file_y.txt";
    const char* world_coordinate_imge_file_z="/Users/jimmy/Desktop/SCRF_RGBD_train/TUM/world_coordinate_imge_file_z.txt";
    const char* world_coordinate_imge_file_depth="/Users/jimmy/Desktop/SCRF_RGBD_train/TUM/depth.txt";
    
    
    CvxIO::save_mat(world_coordinate_imge_file_x, world_coordinate_image_x);
    CvxIO::save_mat(world_coordinate_imge_file_y, world_coordinate_image_y);
    CvxIO::save_mat(world_coordinate_imge_file_z, world_coordinate_image_z);
    CvxIO::save_mat(world_coordinate_imge_file_depth, camera_depth_img);

}


void test_datasetParameter()
{
    const char* dataset_param_filename="/Users/jimmy/Desktop/SCRF_RGBD_train/SCRF_RGBD_Fire/dataset_param_TUM.txt";
    
    DatasetParameter dataset_param;
    
    dataset_param.readFromFileDataParameter(dataset_param_filename);
    
}


void test_rgbgRF_single_tree()
{
   /* char depth_img_file[] = "/Users/jimmy/Desktop/images/7_scenes/heads/seq-02/frame-000085.depth.png";
    char rgb_img_file[]   = "/Users/jimmy/Desktop/images/7_scenes/heads/seq-02/frame-000085.color.png";
    char pose_file[] = "/Users/jimmy/Desktop/images/7_scenes/heads/seq-02/frame-000085.pose.txt";
   
    const char* dataset_param_filename="/Users/jimmy/Desktop/SCRF_RGBD_train/SCRF_RGBD_Fire/dataset_param_7Scenes.txt";
    */
    
    char depth_img_file[] ="/Users/jimmy/Desktop/images/TUM_SLAM/freiburg3_long_office_household/seq-01/frame-000000.depth.png";
    char rgb_img_file[]   = "/Users/jimmy/Desktop/images/TUM_SLAM/freiburg3_long_office_household/seq-01/frame-000000.color.png";
    char pose_file [] = "/Users/jimmy/Desktop/images/TUM_SLAM/freiburg3_long_office_household/seq-01/frame-000000.pose.txt";
    const char* dataset_param_filename="/Users/jimmy/Desktop/SCRF_RGBD_train/SCRF_RGBD_Fire/dataset_param_TUM.txt";
    
    
    cv::Mat camera_depth_img;
    cv::Mat rgb_img;
    bool is_read = CvxIO::imread_depth_16bit_to_64f(depth_img_file, camera_depth_img);
    assert(is_read);
    CvxIO::imread_rgb_8u(rgb_img_file, rgb_img);
    
    DatasetParameter dataset_param;
    
    dataset_param.readFromFileDataParameter(dataset_param_filename);
    double depth_factor = dataset_param.depth_factor_;
    
  
    
    cv::Mat calibration_matrix = cv::Mat::eye(3, 3, CV_64F);
    calibration_matrix.at<double>(0, 0) = dataset_param.k_focal_length_x_;
    calibration_matrix.at<double>(1, 1) = dataset_param.k_focal_length_y_;
    calibration_matrix.at<double>(0, 2) = dataset_param.k_camera_centre_u_;
    calibration_matrix.at<double>(1, 2) = dataset_param.k_camera_centre_v_;
    
    double min_depth = 0.05;
    double max_depth = 10.0;
    
    int num_sample = 5000;
    int image_index = 0;
    
    vector<RGBGLearningSample> samples = RGBGUtil::randomSampleFromRgbdImages(rgb_img_file,
                                                                              depth_img_file,
                                                                                         pose_file,
                                                                                         num_sample,
                                                                                         image_index,
                                                                                         depth_factor,
                                                                                         calibration_matrix,
                                                                                         min_depth,
                                                                                         max_depth, false);
                                                        
    cv::GaussianBlur(rgb_img, rgb_img, cv::Size(5, 5), 0.0);
    
   
    RGBGTree tree;
    RGBGTreeParameter param;
    param.verbose_ = true;
    param.min_leaf_node_ = 2;
    param.max_depth_ = 20;
    
    
    int num_train = num_sample/3*2;
    
    vector<unsigned int> indices;
    for (int i = 0; i<num_train; i++) {
        indices.push_back(i);
    }
    vector<cv::Mat> rgbImages;
    rgbImages.push_back(rgb_img);
    tree.buildTree(samples, indices, rgbImages, param);
    
    vector<RGBGTestingResult> predictions;
    vector<double> distance;
    for (int i = num_train; i<samples.size(); i++) {
        RGBGTestingResult pred;
        bool is_predict = tree.predict(samples[i], rgb_img, pred);
        if (is_predict) {
            predictions.push_back(pred);
            cv::Point3d dif = pred.predict_error;
            cout<<"dif "<<dif<<endl;
            double dis = dif.x * dif.x + dif.y * dif.y + dif.z * dif.z;
            dis = sqrt(dis);
            distance.push_back(dis);
        }
    }
    printf("predicted %lu from %lu, percentage %lf\n", predictions.size(), samples.size() - num_train, 1.0*(predictions.size())/(samples.size() - num_train));
    
    std::sort(distance.begin(), distance.end());
    double median_dis = distance[distance.size()/2];
    cout<<"median distance is "<<median_dis<<endl;
    
    cv::Point3d test_error = RGBGUtil::predictionErrorStddev(predictions);
    cout<<"validation error from read file is "<<test_error<<endl;
}

void test_rgbgRF_single_tree_multi_img()
{
    const char* dataset_param_filename="/Users/jimmy/Desktop/SCRF_RGBD_train/TUM/dataset_param_TUM.txt";
    
    const char* train_depth_file_name ="/Users/jimmy/Desktop/SCRF_RGBD_train/TUM/train/train_depth_list.txt";
    const char* train_rgb_file_name  = "/Users/jimmy/Desktop/SCRF_RGBD_train/TUM/train/train_rgb_list.txt";
    const char* train_pose_file_name  = "/Users/jimmy/Desktop/SCRF_RGBD_train/TUM/train/train_pose_list.txt";
    
    
    DatasetParameter dataset_param;
    
    dataset_param.readFromFileDataParameter(dataset_param_filename);
    double depth_factor = dataset_param.depth_factor_;
    
    
    cv::Mat calibration_matrix = cv::Mat::eye(3, 3, CV_64F);
    calibration_matrix.at<double>(0, 0) = dataset_param.k_focal_length_x_;
    calibration_matrix.at<double>(1, 1) = dataset_param.k_focal_length_y_;
    calibration_matrix.at<double>(0, 2) = dataset_param.k_camera_centre_u_;
    calibration_matrix.at<double>(1, 2) = dataset_param.k_camera_centre_v_;
    
    double min_depth = 0.05;
    double max_depth = 10.0;

    
    vector<string> rgb_files_train  = Ms7ScenesUtil::read_file_names(train_rgb_file_name);
    vector<string> depth_files_train = Ms7ScenesUtil::read_file_names(train_depth_file_name);
    vector<string> pose_files_train  = Ms7ScenesUtil::read_file_names(train_pose_file_name);
    
    
    vector<RGBGLearningSample> all_samples;
    vector<cv::Mat> rgb_images_train;
    
    int num_sample = 5000;
    

    // read rgb files for training
    for (int i = 0; i<rgb_files_train.size(); i++) {
        const char *rgb_img_file     = rgb_files_train[i].c_str();
        const char *depth_img_file   = depth_files_train[i].c_str();
        const char *pose_file        = pose_files_train[i].c_str();
        
        cv::Mat camera_depth_img;
        cv::Mat rgb_img;
        bool is_read = CvxIO::imread_depth_16bit_to_64f(depth_img_file, camera_depth_img);
        assert(is_read);
        CvxIO::imread_rgb_8u(rgb_img_file, rgb_img);
        
        
        vector<RGBGLearningSample> samples = RGBGUtil::randomSampleFromRgbdImages(rgb_img_file,
                                                                                              depth_img_file,
                                                                                              pose_file,
                                                                                              num_sample,
                                                                                              i,
                                                                                              depth_factor,
                                                                                              calibration_matrix,
                                                                                              min_depth,
                                                                                              max_depth,
                                                                                              true);
        
        all_samples.insert(all_samples.end(), samples.begin(), samples.end());
        rgb_images_train.push_back(rgb_img);
    }
    
    printf("train image number is %lu, sample number is %lu\n", rgb_images_train.size(), all_samples.size());
    
    RGBGTree tree;
    RGBGTreeParameter param;
    param.verbose_ = true;
    param.min_leaf_node_ = 2;
    param.max_depth_ = 20;
    param.max_pixel_offset_ = 233;
  
    int num_samples = all_samples.size();
    int num_train = num_samples*2/3;
    
    vector<unsigned int> indices;
    for(int i = 0; i<num_samples; i++){
        indices.push_back(i);
    }
    
    
    random_shuffle(indices.begin(), indices.end());

    
    vector<unsigned int>::const_iterator indices_begin = indices.begin();
    vector<unsigned int>::const_iterator indices_end = indices.begin()+num_train;
    vector<unsigned int> indices_train(indices_begin, indices_end);
   
    tree.buildTree(all_samples, indices_train, rgb_images_train, param);
    
    
    vector<double> distance;
    for(int j = num_train; j<num_samples; j++) {
        int index = indices[j];
        RGBGTestingResult pred;
        bool is_predict = tree.predict(all_samples[index], rgb_images_train[all_samples[index].image_index_], pred);
        if (is_predict) {
            cv::Point3d dif = pred.predict_error;
            cout<<"dif "<<dif<<endl;
            double dis = dif.x * dif.x + dif.y*dif.y + dif.z*dif.z;
            dis = sqrt(dis);
            distance.push_back(dis);
        }
    }
    
    std::sort(distance.begin(), distance.end());
    double median_dis = distance[distance.size()/2];
    cout<<"median distance is "<<median_dis<<endl;
    
}

void test_rgbgRF_single_tree_multi_img_train_test_dif_imgs()
{
    const char* dataset_param_filename="/Users/jimmy/Desktop/SCRF_RGBD_train/TUM/dataset_param_TUM.txt";
    
    const char* train_depth_file_name ="/Users/jimmy/Desktop/SCRF_RGBD_train/TUM/train/depth_image_list_9.txt";
    const char* train_rgb_file_name  = "/Users/jimmy/Desktop/SCRF_RGBD_train/TUM/train/rgb_image_list_9.txt";
    const char* train_pose_file_name  = "/Users/jimmy/Desktop/SCRF_RGBD_train/TUM/train/camera_pose_list_9.txt";
    
    const char* test_depth_file_name ="/Users/jimmy/Desktop/SCRF_RGBD_train/TUM/test/test_depth_list.txt";
    const char* test_rgb_file_name  = "/Users/jimmy/Desktop/SCRF_RGBD_train/TUM/test/test_rgb_list.txt";
    const char* test_pose_file_name  = "/Users/jimmy/Desktop/SCRF_RGBD_train/TUM/test/test_pose_list.txt";
    
    DatasetParameter dataset_param;
    
    dataset_param.readFromFileDataParameter(dataset_param_filename);
    double depth_factor = dataset_param.depth_factor_;
    
    
    cv::Mat calibration_matrix = cv::Mat::eye(3, 3, CV_64F);
    calibration_matrix.at<double>(0, 0) = dataset_param.k_focal_length_x_;
    calibration_matrix.at<double>(1, 1) = dataset_param.k_focal_length_y_;
    calibration_matrix.at<double>(0, 2) = dataset_param.k_camera_centre_u_;
    calibration_matrix.at<double>(1, 2) = dataset_param.k_camera_centre_v_;
    
    double min_depth = 0.05;
    double max_depth = 10.0;
    
    
    vector<string> rgb_files_train  = Ms7ScenesUtil::read_file_names(train_rgb_file_name);
    vector<string> depth_files_train = Ms7ScenesUtil::read_file_names(train_depth_file_name);
    vector<string> pose_files_train  = Ms7ScenesUtil::read_file_names(train_pose_file_name);
    
    vector<string> rgb_files_test  = Ms7ScenesUtil::read_file_names(test_rgb_file_name);
    vector<string> depth_files_test = Ms7ScenesUtil::read_file_names(test_depth_file_name);
    vector<string> pose_files_test  = Ms7ScenesUtil::read_file_names(test_pose_file_name);
    
    
    vector<RGBGLearningSample> all_samples_train, all_samples_test;
    vector<cv::Mat> rgb_images_train, rgb_images_test;
    
    int num_sample = 5000;
    
    
    // read rgb files for training
    for (int i = 0; i<rgb_files_train.size(); i++) {
        const char *rgb_img_file     = rgb_files_train[i].c_str();
        const char *depth_img_file   = depth_files_train[i].c_str();
        const char *pose_file        = pose_files_train[i].c_str();
        
        
        cv::Mat rgb_img;
        CvxIO::imread_rgb_8u(rgb_img_file, rgb_img);
        
        
        vector<RGBGLearningSample> samples = RGBGUtil::randomSampleFromRgbdImages(rgb_img_file,
                                                                                              depth_img_file,
                                                                                              pose_file,
                                                                                              num_sample,
                                                                                              i,
                                                                                              depth_factor,
                                                                                              calibration_matrix,
                                                                                              min_depth,
                                                                                              max_depth, true);
        
        all_samples_train.insert(all_samples_train.end(), samples.begin(), samples.end());
        rgb_images_train.push_back(rgb_img);
    }
    
    printf("train image number is %lu, sample number is %lu\n", rgb_images_train.size(), all_samples_train.size());
    
    RGBGTree tree;
    RGBGTreeParameter param;
    param.verbose_ = true;
    param.min_leaf_node_ = 2;
    param.max_depth_ = 20;
    param.max_pixel_offset_= 233;
    param.is_use_mean_shift_ = false;
    
    int num_samples = all_samples_train.size();
    
    
    vector<unsigned int> indices;
    for(int i = 0; i<num_samples; i++){
        indices.push_back(i);
    }

    
    tree.buildTree(all_samples_train, indices, rgb_images_train, param);
    
    vector<double> distance;
    // read rgb files for testing
    for (int i = 0; i< rgb_files_test.size(); i++) {
        const char *rgb_img_file     = rgb_files_test[i].c_str();
        const char *depth_img_file   = depth_files_train[i].c_str();
        const char *pose_file        = pose_files_train[i].c_str();
        
        
        cv::Mat rgb_img;
        CvxIO::imread_rgb_8u(rgb_img_file, rgb_img);
        
        
        vector<RGBGLearningSample> samples_test = RGBGUtil::randomSampleFromRgbdImages(rgb_img_file,
                                                                                              depth_img_file,
                                                                                              pose_file,
                                                                                              num_sample,
                                                                                              i,
                                                                                              depth_factor,
                                                                                              calibration_matrix,
                                                                                              min_depth,
                                                                                              max_depth, true);
    
       
        for(int j =0 ; j<samples_test.size(); j++) {
            RGBGTestingResult pred;
            bool is_predict = tree.predict(samples_test[j], rgb_img, pred);
            if (is_predict) {
                cv::Point3d dif = pred.predict_error;
                cout<<"dif "<<dif<<endl;
                double dis = dif.x * dif.x + dif.y*dif.y + dif.z*dif.z;
                dis = sqrt(dis);
                distance.push_back(dis);
            }
        }
    }
    
    std::sort(distance.begin(), distance.end());
    double median_dis = distance[distance.size()/2];
    cout<<"median distance is "<<median_dis<<endl;

}



void test_rgbg_multi_tree_multiple_images_model()
{
    const char* dataset_param_filename="/Users/jimmy/Desktop/SCRF_RGBD_train/TUM/dataset_param_TUM.txt";
    
    const char* train_depth_file_name ="/Users/jimmy/Desktop/TrainTestTUM/train_200/depth_image_list.txt";
    const char* train_rgb_file_name  = "/Users/jimmy/Desktop/TrainTestTUM/train_200/rgb_image_list.txt";
    const char* train_pose_file_name  = "/Users/jimmy/Desktop/TrainTestTUM/train_200/camera_pose_list.txt";
    const char* tree_param_file = "/Users/jimmy/Desktop/TrainTestTUM/RF_param_5trees.txt";
    
    
    const char* test_depth_file_name ="/Users/jimmy/Desktop/TrainTestTUM/test_100/depth_image_list.txt";
    const char* test_rgb_file_name  = "/Users/jimmy/Desktop/TrainTestTUM/test_100/rgb_image_list.txt";
    const char* test_pose_file_name  = "/Users/jimmy/Desktop/TrainTestTUM/test_100/camera_pose_list.txt";
    
    
    DatasetParameter dataset_param;
    dataset_param.readFromFileDataParameter(dataset_param_filename);
    
    
    RGBGTreeParameter tree_param;
    bool is_read = RGBGUtil::readTreeParameter(tree_param_file, tree_param);
    assert(is_read);
    if (tree_param.is_use_depth_) {
        printf("Note: depth is used ...................................................\n");
    }
    else {
        printf("Node: depth is Not used ............................................\n");
    }
    

    
    dataset_param.readFromFileDataParameter(dataset_param_filename);
    double depth_factor = dataset_param.depth_factor_;
    
    
    cv::Mat calibration_matrix = cv::Mat::eye(3, 3, CV_64F);
    calibration_matrix.at<double>(0, 0) = dataset_param.k_focal_length_x_;
    calibration_matrix.at<double>(1, 1) = dataset_param.k_focal_length_y_;
    calibration_matrix.at<double>(0, 2) = dataset_param.k_camera_centre_u_;
    calibration_matrix.at<double>(1, 2) = dataset_param.k_camera_centre_v_;
    
    double min_depth = dataset_param.min_depth_;
    double max_depth = dataset_param.max_depth_;
    
    
    vector<string> rgb_files_train  = Ms7ScenesUtil::read_file_names(train_rgb_file_name);
    vector<string> depth_files_train = Ms7ScenesUtil::read_file_names(train_depth_file_name);
    vector<string> pose_files_train  = Ms7ScenesUtil::read_file_names(train_pose_file_name);
    
    vector<string> rgb_files_test  = Ms7ScenesUtil::read_file_names(test_rgb_file_name);
    vector<string> depth_files_test = Ms7ScenesUtil::read_file_names(test_depth_file_name);
    vector<string> pose_files_test  = Ms7ScenesUtil::read_file_names(test_pose_file_name);
    
    
    vector<RGBGLearningSample> all_samples_train, all_samples_test;
    vector<cv::Mat> rgb_images_train, rgb_images_test;
    
    int num_sample = 5000;
    
    
    // read rgb files for training
    for (int i = 0; i<rgb_files_train.size(); i++) {
        const char *rgb_img_file     = rgb_files_train[i].c_str();
        const char *depth_img_file   = depth_files_train[i].c_str();
        const char *pose_file        = pose_files_train[i].c_str();
        
        
        cv::Mat rgb_img;
        CvxIO::imread_rgb_8u(rgb_img_file, rgb_img);
        
        
        vector<RGBGLearningSample> samples = RGBGUtil::randomSampleFromRgbdImages(rgb_img_file,
                                                                                  depth_img_file,
                                                                                  pose_file,
                                                                                  num_sample,
                                                                                   i,
                                                                                   depth_factor,
                                                                                   calibration_matrix,
                                                                                   min_depth,
                                                                                   max_depth, true);
        
        all_samples_train.insert(all_samples_train.end(), samples.begin(), samples.end());
        rgb_images_train.push_back(rgb_img);
    }
    
    printf("train image number is %lu, sample number is %lu\n", rgb_images_train.size(), all_samples_train.size());
    
   
    RGBGTreeParameter param;
    param.verbose_ = true;
    param.min_leaf_node_ = 2;
    param.max_depth_ = 20;
    param.max_pixel_offset_= 233;
    param.is_use_mean_shift_ = false;
    
    int num_samples = all_samples_train.size();
    
    vector<unsigned int> indices;
    for(int i = 0; i<num_samples; i++){
        indices.push_back(i);
    }
    

    RGBGRegressorBuilder builder;
    RGBGRegressor model;
    
    builder.setTreeParameter(param);
    builder.buildModel(model, all_samples_train, rgb_images_train);
    
    
    model.save("RGBG_regressor.txt");
    
    model = RGBGRegressor();
    model.load("RGBG_regressor.txt");
    
    vector<double> distance;
    // read rgb files for testing
    for (int i = 0; i< rgb_files_test.size(); i++) {
        const char *rgb_img_file     = rgb_files_test[i].c_str();
        const char *depth_img_file   = depth_files_train[i].c_str();
        const char *pose_file        = pose_files_train[i].c_str();
        
        
        cv::Mat rgb_img;
        CvxIO::imread_rgb_8u(rgb_img_file, rgb_img);
        
        
        vector<RGBGLearningSample> samples_test = RGBGUtil::randomSampleFromRgbdImages(rgb_img_file,
                                                                                                   depth_img_file,
                                                                                                   pose_file,
                                                                                                   num_sample,
                                                                                                   i,
                                                                                                   depth_factor,
                                                                                                   calibration_matrix,
                                                                                                   min_depth,
                                                                                                   max_depth, true);
        
        
        for(int j =0 ; j<samples_test.size(); j++) {
            RGBGTestingResult pred;
            bool is_predict = model.predict(samples_test[j], rgb_img, pred);  // replace tree with model as "tree" is Not trained.
            if (is_predict) {
                cv::Point3d dif = pred.predict_error;
                cout<<"dif "<<dif<<endl;
                double dis = dif.x * dif.x + dif.y*dif.y + dif.z*dif.z;
                dis = sqrt(dis);
                distance.push_back(dis);
            }
        }
    }
    
    std::sort(distance.begin(), distance.end());
    double median_dis = distance[distance.size()/2];
    cout<<"median distance is "<<median_dis<<endl;
    
}

void test_4_scenes_angular_dis_within_min_trans_dis()
{
    ofstream fout1("/Users/jimmy/Desktop/RGBD_4Scenes/kitchen/rotDiswithMinTransError_kitchen.txt");
    ofstream fout2("/Users/jimmy/Desktop/RGBD_4Scenes/kitchen/minTransError_kitchen.txt");
    ///for fire
    const char* train_file_name="/Users/jimmy/Desktop/RGBD_4Scenes/kitchen/train_files/camera_pose_list.txt";
    const char* test_file_name = "/Users/jimmy/Desktop/RGBD_4Scenes/kitchen/test_files/camera_pose_list.txt";
    
    
    vector<double> rot_dis_under_min_trans_dis, min_trans_dis_vec;
    
    camera_rot_dis_under_min_trans_dis(train_file_name,
                                       test_file_name,
                                       min_trans_dis_vec,
                                       rot_dis_under_min_trans_dis);
    
    assert(min_trans_dis_vec.size()==rot_dis_under_min_trans_dis.size());
    cout<<rot_dis_under_min_trans_dis.size()<<endl;
    cout<<min_trans_dis_vec.size()<<endl;
    for(int i=0; i<rot_dis_under_min_trans_dis.size();i++)
    {
        
        fout1<<rot_dis_under_min_trans_dis[i]<<endl;
        fout2<<min_trans_dis_vec[i]<<endl;
    }

}
