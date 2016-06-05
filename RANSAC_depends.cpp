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


/*
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

}*/

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


struct HypotheseLoss
{
    double loss_;
    Mat rvec_;    //rotation
    Mat tvec_;    //translation
    Eigen::Matrix3Xd input;
    Eigen::Matrix3Xd output;
    Eigen::Vector3d camera_space_point;
    Eigen::Affine3d pose;

    vector<unsigned int> inlier_indices_;

    HypotheseLoss()
    {
        loss_ = INT_MAX;
    }

    HypotheseLoss(const double loss)
    {
        loss_ = loss;
    }

    HypotheseLoss(const HypotheseLoss & other)
    {
        loss_ = other.loss_;
        rvec_ = other.rvec_;
        tvec_ = other.tvec_;
        inlier_indices_.clear();
        inlier_indices_.resize(other.inlier_indices_.size());

        for(int i=0; i < other.inlier_indices_.size(); i++)
        {
            inlier_indices_[i] = other.inlier_indices_[i];
        }
    }

    bool operator < (const HypotheseLoss & other) const
    {
        return loss_ < other.loss_;
    }

    HypotheseLoss & operator = (const HypotheseLoss & other)
    {
        if(&other == this)
        {
            return *this;
        }

        loss_ = other.loss_;
        rvec_ = other.rvec_;
        tvec_ = other.tvec_;
        inlier_indices_.clear();
        inlier_indices_.resize(other.inlier_indices_.size());

        for(int i=0; i<other.inlier_indices_.size(); i++)
        {
            inlier_indices_[i] = other.inlier_indices_[i];
        }

        return *this;
    }

};


struct PreemptiveRANSACParameter
{
    double reproj_threshold;

public:
    PreemptiveRANSACParameter()
    {
        reproj_threshold = 4.0;
    }
};


bool preemptiveRANSAC(const vector<cv::Point3d> & wld_pts,
                      const vector<cv::Point2d> & img_pts,
                      const cv::Mat & camera_matrix,
                      const cv::Mat & dist_coeff,
                      const PreemptiveRANSACParameter & param,
                      cv::Mat & camera_pose)
{
    assert(img_pts.size() == wld_pts.size());
    assert(img_pts.size() > 500);

    const int num_iteration = 1000000;
    int K = 1024;
    const int N = (int)img_pts.size();
    const int B = 500;

    vector<std::pair<Mat, Mat> > rt_candidate;

    for(int i=0; i<num_iteration; i++)
    {
        cout<<i<<endl;
        int k1 = 0;
        int k2 = 0;
        int k3 = 0;
        int k4 = 0;

        do{
            k1 = rand()%N;
            k2 = rand()%N;
            k3 = rand()%N;
            k4 = rand()%N;
        } while(k1 == k2 || k1 == k3 || k1==k4 ||
                k2 == k3 || k2 == k4 || k3==k4);

        vector<cv::Point2d> sampled_img_pts;
        vector<cv::Point3d> sampled_wld_pts;

        sampled_img_pts.push_back(img_pts[k1]);
        sampled_img_pts.push_back(img_pts[k2]);
        sampled_img_pts.push_back(img_pts[k3]);

        sampled_wld_pts.push_back(wld_pts[k1]);
        sampled_wld_pts.push_back(wld_pts[k2]);
        sampled_wld_pts.push_back(wld_pts[k3]);

        Mat rvec;
        Mat tvec;

        bool is_solved = cv::solvePnP(Mat(sampled_wld_pts), Mat(sampled_img_pts), camera_matrix, dist_coeff, rvec, tvec, false, CV_EPNP);

        if(is_solved)
        {
            rt_candidate.push_back(std::make_pair(rvec, tvec));
        }

        /*
        if(rt_candidate.size()> K)
        {
           // printf("initialization repeat %d times\n", i);
            break;
        }*/
    }

    printf("init camera parameter number is %lu\n", rt_candidate.size());

    K = (int)rt_candidate.size();

    vector<HypotheseLoss> losses;
    for(int i = 0; i< rt_candidate.size(); i++)
    {
        HypotheseLoss hyp(0.0);
        hyp.rvec_ = rt_candidate[i].first;
        hyp.tvec_ = rt_candidate[i].second;
        losses.push_back(hyp);
    }

    double reproj_threshold = param.reproj_threshold;
    while( losses.size()>1 )
    {
        //sample random set
        vector<cv::Point2d> sampled_img_pts;
        vector<cv::Point3d> sampled_wld_pts;

        for(int i=0; i<B; i++)
        {
            int index = rand()%N;
            sampled_img_pts.push_back(img_pts[index]);
            sampled_wld_pts.push_back(wld_pts[index]);
        }

        //count outliers
        cout<<"losses.size() is "<<losses.size()<<endl;

        for(int i=0; i<losses.size(); i++)
        {
            //evaluate the accuracy by check reprojection error
            vector<cv::Point2d> projected_pts;

            cv::projectPoints(sampled_wld_pts, losses[i].rvec_, losses[i].tvec_, camera_matrix, dist_coeff, projected_pts);


            for(int j=0; j<projected_pts.size(); j++)
            {
                cv::Point2d dif = projected_pts[j] - sampled_img_pts[j];
                double dis = cv::norm(dif);

                if(dis > reproj_threshold)
                {
                    losses[i].loss_ +=1.0;
                }
                else
                {
                    losses[i].inlier_indices_.push_back(j);
                }
            }
        }


        std::sort(losses.begin(), losses.end());
        losses.resize(losses.size()/2);

        for(int j=0; j<losses.size(); j++)
        {
            printf("after: loss is %lf\n", losses[j].loss_);
        }

        printf("\n\n");

        //refine inliers
        for(int i=0; i<losses.size(); i++)
        {
            //number of inliers is larger than minimum configuration
            if(losses[i].inlier_indices_.size()>4)
            {
                vector<cv::Point2d> inlier_img_pts;
                vector<cv::Point3d> inlier_wld_pts;

                for(int j = 0; j<losses[i].inlier_indices_.size(); j++)
                {
                    int index = losses[i].inlier_indices_[j];
                    inlier_img_pts.push_back(sampled_img_pts[index]);
                    inlier_wld_pts.push_back(sampled_wld_pts[index]);
                }

                Mat rvec = losses[i].rvec_;
                Mat tvec = losses[i].tvec_;

                bool is_solved = cv::solvePnP(Mat(inlier_wld_pts), Mat(inlier_img_pts), camera_matrix, dist_coeff, rvec, tvec, true, CV_EPNP);
                if(is_solved)
                {
                    losses[i].rvec_ = rvec;
                    losses[i].tvec_ = tvec;
                }
            }
        }
    }

    assert(losses.size() == 1);

    // change to camera to world transformation
    Mat rot;
    cv::Rodrigues(losses.front().rvec_, rot);
    Mat tvec = losses.front().tvec_;
    camera_pose = cv::Mat::eye(4, 4, CV_64F);

    for(int j=0; j<3; j++)
    {
        for(int i = 0; i<3; i++)
        {
            camera_pose.at<double>(i, j) = rot.at<double>(i, j);
        }
    }

    camera_pose.at<double>(0, 3) = tvec.at<double>(0, 0);
    camera_pose.at<double>(1, 3) = tvec.at<double>(1, 0);
    camera_pose.at<double>(2, 3) = tvec.at<double>(2, 0);

    // camera to world coordinate
    camera_pose = camera_pose.inv();

    return true;
}

int main()
{
     ofstream fout("/home/lili/PatternRecognition/RANSAC/chess_inliers_and_error/980/estimated_poses_error980.txt");

     ofstream fout2("/home/lili/PatternRecognition/RANSAC/chess_inliers_and_error/980/inliers_num980.txt");

    string rgb_img_file ="/home/lili/BMVC/7_scenes/chess/seq-03/frame-000980.color.png";
    string depth_img_file = "/home/lili/BMVC/7_scenes/chess/seq-03/frame-000980.depth.png";
    string camera_pose_file = "/home/lili/BMVC/7_scenes/chess/seq-03/frame-000980.pose.txt";

    vector<cv::Point2d> img_pts;
    vector<cv::Point3d> wld_pts_pred;
    vector<cv::Point3d> wld_pts_gt;
    vector<cv::Vec3d> color_pred;
    vector<cv::Vec3d> color_sample;
    const char* leaf_file_name="/home/lili/PatternRecognition/RANSAC/chess_inliers_and_error/980/rgb4000_small_leaf_node_000003.txt";

    ms_7_scenes_util::load_prediction_result_with_color(leaf_file_name,
                                      rgb_img_file,
                                      depth_img_file,
                                      camera_pose_file,
                                      img_pts,
                                      wld_pts_pred,
                                      wld_pts_gt,
                                      color_pred,
                                      color_sample);

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

     */
     string data_loc="/home/lili/PatternRecognition/RANSAC/chess_inliers_and_error/980/rgb4000_small_leaf_node_000003_previous.txt";
     const char* pose_loc="/home/lili/BMVC/7_scenes/chess/seq-03/frame-000980.pose.txt";
    // readData getData(data_loc);
     readPose gtPose;


     int N=wld_pts_gt.size();
     cout<<"The number of points is "<<N<<endl;


     Mat gt_pose=gtPose.getPose(pose_loc);
     cout<<"ground truth pose is "<<gt_pose<<endl;

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

    /*
    ///rotation error between ground truth and the predicted data
    Eigen::Matrix3d gt_rot;
    gt_rot<< gt_pose.at<double>(0,0), gt_pose.at<double>(0,1), gt_pose.at<double>(0,2),
             gt_pose.at<double>(1,0), gt_pose.at<double>(1,1), gt_pose.at<double>(1,2),
             gt_pose.at<double>(2,0), gt_pose.at<double>(2,1), gt_pose.at<double>(2,2);


    Eigen::Quaterniond gt_quater(gt_rot);

    Eigen::Matrix3d pred_rot;
    pred_rot<<camera_pose.at<double>(0,0), camera_pose.at<double>(0,1), camera_pose.at<double>(0,2),
              camera_pose.at<double>(1,0), camera_pose.at<double>(1,1), camera_pose.at<double>(1,2),
              camera_pose.at<double>(2,0), camera_pose.at<double>(2,1), camera_pose.at<double>(2,2);

    Eigen::Quaterniond pred_quater(pred_rot);

    gt_quater.normalize();
    pred_quater.normalize();

    double val_dot = fabs(gt_quater.dot(pred_quater));
    double error_rot = 2.0 *acos(val_dot)*180.0 / M_PI;
    //printf("rotation error is %lf degrees\n", error_rot);

    ///Translation error between ground truth and the predicted data
    Eigen::Vector3d gt_trans(3);
    gt_trans<<gt_pose.at<double>(0,3), gt_pose.at<double>(1,3), gt_pose.at<double>(2,3);

    Eigen::Vector3d pred_trans(3);
    pred_trans<<camera_pose.at<double>(0,3), camera_pose.at<double>(1,3), camera_pose.at<double>(2,3);

    Eigen::Vector3d error_trans_3d=gt_trans-pred_trans;


    double error_trans=error_trans_3d.norm();
    //printf("translation error is %lf m\n", error_trans);
    */
    double error_rot = 0.0;
    double error_trans = 0.0;

    CvxPoseEstimation::poseDistance(camera_pose,
                      gt_pose,
                      error_rot,
                      error_trans);

    trans_error_vec.push_back(error_trans);
    rot_error_vec.push_back(error_rot);

    fout<<error_trans<<" "<<error_rot<<endl;

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


   // std::sort (inliers_vec.begin(), inliers_vec.end(), mycompare);

    for(int i=0; i<10; i++)
    {

        cout<<"top "<<i<<" "<<"inliers_num "<<inliers_vec[sorted_index_vec[i]]<<" inlier number percentage(reprojection_error<10) is "<<1.0*inliers_vec[sorted_index_vec[i]]/img_pts.size()<<endl;
        fout2<<i<<" "<<inliers_vec[sorted_index_vec[i]]<<" "<<1.0*inliers_vec[sorted_index_vec[i]]/img_pts.size()<<" "<<trans_error_vec[sorted_index_vec[i]]<<" "<<rot_error_vec[sorted_index_vec[i]]<<endl;

    }



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

    return 0;

}
