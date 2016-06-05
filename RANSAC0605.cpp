//Author Lili Meng
//All rights reserved

#include <stdio.h>
#include "cvxImage_310.hpp"
#include "cvxPoseEstimation.hpp"
#include <vector>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "RANSAC.h"
#include "cvx_io.hpp"
#include "cvxIO.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>

using std::vector;

using namespace std;

using namespace cv;

struct mycomparision {
  bool operator() (int i,int j) { return (i>j);}
} mycompare;

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

    //initialize original index locations
    vector<size_t> index(v.size());
    for(size_t i=0; i !=index.size(); ++i)
    {
        index[i]=i;
    }

    //sort indexes based on comparing values in v
    sort(index.begin(), index.end(), [&v](size_t i1, size_t i2) {return v[i1] > v[i2]; } );

    return index;

}



readData::readData(string filename)
{
    std::ifstream fin(filename.c_str(), std::ios::in);
    if(!fin.is_open())
    {
        cout<<"cannot open file"<<endl;
    }

    istringstream istr;

    double oneDimension;
    vector<double> dataPointVec;
    string str;

     while(getline(fin,str))
     {
        istr.str(str);
        while(istr>>oneDimension)
        {
            dataPointVec.push_back(oneDimension);
        }
        allDataPointsVec.push_back(dataPointVec);
        dataPointVec.clear();
        istr.clear();
        str.clear();
     }
     fin.close();

    int numOfDimensions=allDataPointsVec[0].size();
    int numOfElements=allDataPointsVec.size();

    cout<<"The number of points is "<<numOfElements<<endl;

    for(int i=0; i<numOfElements; i++)
    {
        //cout<<"PointID is "<<i<<" "<<j<<"t"<<setprecision(20)<<"value is "<<allDataPointsVec[i][j]<<endl;
        img_pts.push_back(Point2d(allDataPointsVec[i][0],allDataPointsVec[i][1]));
        pred_wld_pts.push_back(Point3d(allDataPointsVec[i][2],allDataPointsVec[i][3],allDataPointsVec[i][4]));
        gt_wld_pts.push_back(Point3d(allDataPointsVec[i][5],allDataPointsVec[i][6],allDataPointsVec[i][7]));
        pred_color.push_back(Point3d(allDataPointsVec[i][8],allDataPointsVec[i][9],allDataPointsVec[i][10]));
        actual_color.push_back(Point3d(allDataPointsVec[i][11],allDataPointsVec[i][12],allDataPointsVec[i][13]));

    }

}

readPose::readPose(){};

Mat readPose::getPose(const char *file_name)
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



int main()
{

    // ofstream fout("/home/lili/PatternRecognition/RANSAC/chess_inliers_result/estimated_poses_error980.txt");

     //ofstream fout2("/home/lili/PatternRecognition/RANSAC/chess_inliers_result/inliers_num980.txt");

     ofstream fout3("/home/lili/PatternRecognition/RANSAC/chess_inliers_result/inliers_tran_rot_error_200images.txt");

    /*
    string rgb_img_file ="/home/lili/BMVC/7_scenes/chess/seq-03/frame-000980.color.png";
    string depth_img_file = "/home/lili/BMVC/7_scenes/chess/seq-03/frame-000980.depth.png";
    string camera_pose_file = "/home/lili/BMVC/7_scenes/chess/seq-03/frame-000980.pose.txt";
    */

    vector<double> inliers_all_imgs;
    vector<double> trans_error_all_imgs;
    vector<double> rot_error_all_imgs;


    //const char* leaf_file_name="/home/lili/PatternRecognition/RANSAC/chess_inliers_and_error/980/rgb4000_small_leaf_node_000003.txt";

    const char* leaf_file_dir="/home/lili/PatternRecognition/chess_prediction_result/predict_result/*.txt";

    vector<string> leaf_files_vec=CvxIO::read_files(leaf_file_dir);

    int num_imgs=(int)leaf_files_vec.size();
    cout<<num_imgs<<endl;
    cout<<leaf_files_vec[0]<<endl;

    vector<vector<cv::Point2d> > img_pts_vec;
    vector<vector<cv::Point3d> > wld_pts_pred_vec;
    vector<vector<cv::Point3d> > wld_pts_gt_vec;
    vector<vector<cv::Vec3d> > color_pred_vec;
    vector<vector<cv::Vec3d> > color_sample_vec;


    string img_sub_dir="/home/lili/BMVC/";
    vector<string> rgb_imgs_vec;
    vector<string> depth_imgs_vec;
    vector<string> camera_poses_vec;


    for(int m=0; m<1; m++)
    {

        string rgb_img_file_jimmy, depth_img_file_jimmy, camera_pose_file_jimmy;

        vector<cv::Point2d> img_pts;
        vector<cv::Point3d> wld_pts_pred;
        vector<cv::Point3d> wld_pts_gt;
        vector<cv::Vec3d> color_pred;
        vector<cv::Vec3d> color_sample;

        ms_7_scenes_util::load_prediction_result_with_color(leaf_files_vec[m].c_str(),
                                      rgb_img_file_jimmy,
                                      depth_img_file_jimmy,
                                      camera_pose_file_jimmy,
                                      img_pts,
                                      wld_pts_pred,
                                      wld_pts_gt,
                                      color_pred,
                                      color_sample);

        cout<<"rgb_img_file_jimmy "<<rgb_img_file_jimmy<<endl;
        cout<<"depth_img_file_jimmy "<<depth_img_file_jimmy<<endl;
        cout<<"camera_pose_file_jimmy "<<camera_pose_file_jimmy<<endl;

        /// find the correct directory from lili's computer
        size_t rgb_substr_dir_pos = rgb_img_file_jimmy.find("7_scenes");
        string rgb_substr_dir = rgb_img_file_jimmy.substr(rgb_substr_dir_pos);
        string rgb_img_file = img_sub_dir+rgb_substr_dir;

        size_t depth_substr_dir_pos= depth_img_file_jimmy.find("7_scenes");
        string depth_substr_dir = depth_img_file_jimmy.substr(depth_substr_dir_pos);
        string depth_img_file = img_sub_dir+ depth_substr_dir;


        size_t pose_substr_dir_pos= camera_pose_file_jimmy.find("7_scenes");
        string pose_substr_dir = camera_pose_file_jimmy.substr(pose_substr_dir_pos);
        string camera_pose_file = img_sub_dir+ pose_substr_dir;
        cout<<"camera_pose_file now "<<camera_pose_file<<endl;


        ///find the frame number
        size_t frame_num_pos=rgb_img_file_jimmy.find("frame");
        string frame_num= rgb_img_file_jimmy.substr(frame_num_pos);

        /*
        cout<<rgb_substr_dir<<endl;
        cout<<rgb_img_file<<endl;
        cout<<depth_img_file<<endl;
        cout<<camera_pose_file<<endl;
        */

        rgb_imgs_vec.push_back(rgb_img_file);
        depth_imgs_vec.push_back(depth_img_file);
        camera_poses_vec.push_back(camera_pose_file);

        assert(rgb_imgs_vec.size() == depth_imgs_vec.size());
        assert(rgb_imgs_vec.size() == camera_poses_vec.size());


     /*
     string rgb_loc="/home/lili/BMVC/7_scenes/chess/seq-03/frame-000980.color.png";
     Mat rgb_frame=imread(rgb_loc,  CV_LOAD_IMAGE_COLOR);
     assert(rgb_frame.type()==CV_8UC3);
    // imwrite("/home/lili/PatternRecognition/RANSAC/chess_inliers_and_error/980/rgb_980.png",rgb_frame);

     imshow("RGB",rgb_frame);

     string depth_loc="/home/lili/BMVC/7_scenes/chess/seq-03/frame-000980.depth.png";
     Mat depth_frame=imread(depth_loc, CV_LOAD_IMAGE_ANYDEPTH);
     assert(depth_frame.type() == CV_16UC1);
     depth_frame.convertTo(depth_frame, CV_64F);
     //imshow("Depth",depth_frame);
     //waitKey(0);


     string data_loc="/home/lili/PatternRecognition/RANSAC/chess_inliers_and_error/980/rgb4000_small_leaf_node_000003_previous.txt";
     const char* pose_loc="/home/lili/BMVC/7_scenes/chess/seq-03/frame-000980.pose.txt";

     */
     //readData getData(rgb_img_file);
     readPose gtPose;


     int N=wld_pts_gt.size();
     cout<<"The number of points is "<<N<<endl;

     const char* pose_loc=camera_pose_file.c_str();
     cout<<pose_loc<<endl;
     Mat gt_pose=gtPose.getPose(pose_loc);


    //cout<<"ground truth pose is "<<gt_pose<<endl;

     Mat rvec;
     Mat tvec;

    double cx= 320;
    double cy= 240;
    double fx= 585;
    double fy= 585;

    cv::Mat camera_matrix(3,3,CV_64F);

    camera_matrix.at<double>(0,0)=fx;
    camera_matrix.at<double>(0,1)=0;
    camera_matrix.at<double>(0,2)=cx;
    camera_matrix.at<double>(1,0)=0;
    camera_matrix.at<double>(1,1)=fy;
    camera_matrix.at<double>(1,2)=cy;
    camera_matrix.at<double>(2,0)=0;
    camera_matrix.at<double>(2,1)=0;
    camera_matrix.at<double>(2,2)=1;

    //cv::solvePnPRansac(Mat(getData.pred_wld_pts), Mat(getData.img_pts), camera_matrix, Mat(), rvec, tvec, false, 1000, 8.0);
    //cv::solvePnP(Mat(getData.pred_wld_pts), Mat(getData.img_pts), camera_matrix, Mat(), rvec, tvec, false, CV_EPNP);



    int iteration_num=1000;
    vector<double> trans_error_vec;
    vector<double> rot_error_vec;
    vector<int> inliers_vec;

    for(int i=0; i<iteration_num; i++)
    {

        //sample random set
        vector<cv::Point2d> four_img_pts;
        vector<cv::Point3d> four_wld_pts;

        for(int j=0; j<4; j++)
        {
            int index = rand()%N;
            four_img_pts.push_back(img_pts[index]);
            four_wld_pts.push_back(wld_pts_pred[index]);
        }


        cv::solvePnP(Mat(four_wld_pts), Mat(four_img_pts), camera_matrix, Mat(), rvec, tvec, false, CV_P3P);


        const PreemptiveRANSACParameter param;
        cv::Mat camera_pose = cv::Mat::eye(4, 4, CV_64F);


        //preemptiveRANSAC(Mat(getData.pred_wld_pts), Mat(getData.img_pts), camera_matrix, Mat(), param, camera_pose);


        //cv::solvePnPRansac(Mat(getData.pred_wld_pts), Mat(getData.img_pts), camera_matrix, Mat(), rvec, tvec, false, 1000, 8.0);
        // change to camera to world transformation


    Mat rot;
    cv::Rodrigues(rvec, rot);

    for(int j=0; j<3; j++)
    {
        for(int k = 0; k<3; k++)
        {
            camera_pose.at<double>(j, k) = rot.at<double>(j, k);
        }
    }

    camera_pose.at<double>(0, 3) = tvec.at<double>(0, 0);
    camera_pose.at<double>(1, 3) = tvec.at<double>(1, 0);
    camera_pose.at<double>(2, 3) = tvec.at<double>(2, 0);


    // camera to world coordinate
    camera_pose = camera_pose.inv();


    double error_rot = 0.0;
    double error_trans = 0.0;

    CvxPoseEstimation::poseDistance(camera_pose,
                      gt_pose,
                      error_rot,
                      error_trans);

    trans_error_vec.push_back(error_trans);
    rot_error_vec.push_back(error_rot);

//    fout<<error_trans<<" "<<error_rot<<endl;

    vector<cv::Point2d> projected_pts;

    cv::projectPoints(Mat(wld_pts_pred), rvec, tvec, camera_matrix, Mat(), projected_pts);
    assert(img_pts.size()==projected_pts.size());


    int inlier_count=0;
    for(int j=0; j<projected_pts.size(); j++)
    {
        double error_reproj = cv::norm(img_pts[j]-projected_pts[j]);

        if(error_reproj<10)
        {
            inlier_count++;
        }

    }

    inliers_vec.push_back(inlier_count);

    } // end of i

    vector<int>::iterator max_inliers_ite=std::max_element(std::begin(inliers_vec), std::end(inliers_vec));

    int max_inliers_position=std::distance(std::begin(inliers_vec), max_inliers_ite);

    int max_inliers_num = inliers_vec[max_inliers_position];

    cout<<"max_inliers_position "<<max_inliers_position<<"max_liers_num "<<max_inliers_num<<" inlier number percentage(reprojection_error<10) is "<<1.0*max_inliers_num/img_pts.size()<<endl;

    vector<size_t> sorted_index_vec=sort_indexes(inliers_vec);

    double trans_error_at_max_inliers=trans_error_vec[max_inliers_position];
    double rot_error_at_max_inliers=rot_error_vec[max_inliers_position];

    cout<<"trans_error_at_max_inliers "<<trans_error_at_max_inliers<<endl;
    cout<<"rot_error_at_max_inliers "<<rot_error_at_max_inliers<<endl;

    fout3<<frame_num<<" "<<max_inliers_num<<" "<<trans_error_at_max_inliers<<" "<<rot_error_at_max_inliers<<endl;

  // std::sort (inliers_vec.begin(), inliers_vec.end(), mycompare);

   /*
    for(int i=0; i<1; i++)
    {

        //cout<<"top "<<i<<" "<<"inliers_num "<<inliers_vec[sorted_index_vec[i]]<<" inlier number percentage(reprojection_error<10) is "<<1.0*inliers_vec[sorted_index_vec[i]]/img_pts.size()<<endl;
        //fout2<<i<<" "<<inliers_vec[sorted_index_vec[i]]<<" "<<1.0*inliers_vec[sorted_index_vec[i]]/img_pts.size()<<" "<<trans_error_vec[sorted_index_vec[i]]<<" "<<rot_error_vec[sorted_index_vec[i]]<<endl;

        cout<<i<<" "<<inliers_vec[sorted_index_vec[i]]<<" "<<1.0*inliers_vec[sorted_index_vec[i]]/img_pts.size()<<" "<<trans_error_vec[sorted_index_vec[i]]<<" "<<rot_error_vec[sorted_index_vec[i]]<<endl;

    }
    */



   /*
    vector<double>::iterator min_trans_ite=std::min_element(std::begin(trans_error_vec),std::end(trans_error_vec));

    int min_trans_position=std::distance(std::begin(trans_error_vec), min_trans_ite);

    double min_trans_error=trans_error_vec[min_trans_position];
    double rot_at_min_trans_error=rot_error_vec[min_trans_position];

    cout<<"min_trans_error  "<<min_trans_error<<" m"<<" at position "<<min_trans_position<<" for this position, rot_error is "<<rot_at_min_trans_error<<" degrees"<<endl;


    double min_rot_error=*std::min_element(std::begin(rot_error_vec),std::end(rot_error_vec));
    vector<double>::iterator min_rot_ite=std::min_element(std::begin(rot_error_vec),std::end(rot_error_vec));
    int min_rot_position=std::distance(std::begin(rot_error_vec), min_rot_ite);

    double trans_at_min_rot_error=trans_error_vec[min_rot_position];

    cout<<"min_rot_error  "<<min_rot_error<<" degrees."<<" at position "<<min_rot_position<<" for this position, trans_error is "<<trans_at_min_rot_error<<" m"<<endl;
    */

    }

    return 0;

}
