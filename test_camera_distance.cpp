//
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

/*void camera_rot_distance(const char* train_file_name,
                         const char* test_file_name,
                         const double trans_threshold,
                         vector<double> &test_min_rot_vec_within_trans_threshold)
{
    vector<string> train_pose_files = Ms7ScenesUtil::read_file_names(train_file_name);
    vector<string> test_pose_files = Ms7ScenesUtil::read_file_names);
    

}*/


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
    
    //for(int k=0; k<test_poses_vec.size(); k++)
    for(int k=0; k<1; k++)
    {
        cv::Mat pose1=test_poses_vec[1];
        cout<<"the pose file from the test sequence is "<<test_pose_files[1]<<endl;
    
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
                cout<<"rot_error is "<<rot_error<<endl;
                cout<<"trans_error is "<<trans_error<<endl;
                //cout<<"within a threshold of "<<rot_threshold<<" the pose file from the training is: "<<train_pose_files[i]<<endl;
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

void test_camera_distance()
{
    ofstream fout("/Users/jimmy/Desktop/RGBTrainChess/test_trans_error10.txt");

    // read ground truth data
    const char* train_file_name="/Users/jimmy/Desktop/RGBTrainChess/train_4000_chess/camera_pose_list.txt";
    const char* test_file_name = "/Users/jimmy/Desktop/RGBTrainChess/test_2000_chess/camera_pose_list.txt";
   
    double rot_threshold=5;
    
    vector<double> test_min_trans_vec_within_rot_threshold;
    camera_trans_distance(train_file_name,
                          test_file_name,
                          rot_threshold,
                          test_min_trans_vec_within_rot_threshold);
    
    for(int i=0; i<test_min_trans_vec_within_rot_threshold.size();i++)
    {
        cout<<test_min_trans_vec_within_rot_threshold[i]<<endl;
        fout<<test_min_trans_vec_within_rot_threshold[i]<<endl;
    }
    
    
}
