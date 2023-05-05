/*
    @file groundplanfit.cpp
    @brief ROS Node for ground plane fitting

    This is a ROS node to perform ground plan fitting.
    Implementation accoriding to <Fast Segmentation of 3D Point Clouds: A Paradigm>

    In this case, it's assumed that the x,y axis points at sea-level,
    and z-axis points up. The sort of height is based on the Z-axis value.

    @author Vincent Cheung(VincentCheungm)
    @bug Sometimes the plane is not fit.
*/

#include <iostream>
// For disable PCL complile lib, to use PointXYZIR    
#define PCL_NO_PRECOMPILE

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>

#include <velodyne_pointcloud/point_types.h>

#include <visualization_msgs/Marker.h>
#include "kitti_loader.hpp"
#include "utils.hpp"

using PointType = PointXYZILID;


// using eigen lib
#include <Eigen/Dense>
using Eigen::MatrixXf;
using Eigen::JacobiSVD;
using Eigen::VectorXf;
using namespace std;

pcl::PointCloud<PointType>::Ptr g_seeds_pc(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr g_ground_pc(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr g_not_ground_pc(new pcl::PointCloud<PointType>());

ros::Publisher CloudPublisher;
ros::Publisher TPPublisher;
ros::Publisher FPPublisher;
ros::Publisher FNPublisher;
ros::Publisher PrecisionPublisher;
ros::Publisher RecallPublisher;


int sensor_model_;
double sensor_height_;
int num_seg_;
int num_iter_;
int num_lpr_;
double th_seeds_;
double th_dist_;

float d_;
MatrixXf normal_;
float th_dist_d_;

bool save_flag;
bool save_csv;

string pcd_savepath;
string data_path;
string output_filename;
string output_gtpoints_name;

void estimate_plane_(void);
void extract_initial_seeds_(const pcl::PointCloud<PointType>& p_sorted);
/*
    @brief Compare function to sort points. Here use z axis.
    @return z-axis accent
*/
bool point_cmp(PointType a, PointType b){
    return a.z<b.z;
}
 

/*
    @brief The function to estimate plane model. The
    model parameter `normal_` and `d_`, and `th_dist_d_`
    is set here.
    The main step is performed SVD(UAV) on covariance matrix.
    Taking the sigular vector in U matrix according to the smallest
    sigular value in A, as the `normal_`. `d_` is then calculated 
    according to mean ground points.

    @param g_ground_pc:global ground pointcloud ptr.
    
*/

void pub_score(std::string mode, double measure) {
    static int                 SCALE = 5;
    visualization_msgs::Marker marker;
    marker.header.frame_id                  = "map";
    marker.header.stamp                     = ros::Time();
    marker.ns                               = "my_namespace";
    marker.id                               = 0;
    marker.type                             = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action                           = visualization_msgs::Marker::ADD;
    if (mode == "p") marker.pose.position.x = 28.5;
    if (mode == "r") marker.pose.position.x = 25;
    marker.pose.position.y                  = 30;

    marker.pose.position.z    = 1;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x            = SCALE;
    marker.scale.y            = SCALE;
    marker.scale.z            = SCALE;
    marker.color.a            = 1.0; // Don't forget to set the alpha!
    marker.color.r            = 0.0;
    marker.color.g            = 1.0;
    marker.color.b            = 0.0;
    //only if using a MESH_RESOURCE marker type:
    marker.text               = mode + ": " + std::to_string(measure);
    if (mode == "p") PrecisionPublisher.publish(marker);
    if (mode == "r") RecallPublisher.publish(marker);

}

template<typename T>
pcl::PointCloud<T> cloudmsg2cloud(sensor_msgs::PointCloud2 cloudmsg) {
    pcl::PointCloud<T> cloudresult;
    pcl::fromROSMsg(cloudmsg, cloudresult);
    return cloudresult;
}

template<typename T>
sensor_msgs::PointCloud2 cloud2msg(pcl::PointCloud<T> cloud, std::string frame_id = "map") {
    sensor_msgs::PointCloud2 cloud_ROS;
    pcl::toROSMsg(cloud, cloud_ROS);
    cloud_ROS.header.frame_id = frame_id;
    return cloud_ROS;
}



void estimate_plane_(void){
    // Create covarian matrix in single pass.
    // TODO: compare the efficiency.
    Eigen::Matrix3f cov;
    Eigen::Vector4f pc_mean;
    pcl::computeMeanAndCovarianceMatrix(*g_ground_pc, cov, pc_mean);
    // Singular Value Decomposition: SVD
    JacobiSVD<MatrixXf> svd(cov,Eigen::DecompositionOptions::ComputeFullU);
    // use the least singular vector as normal
    normal_ = (svd.matrixU().col(2));
    // mean ground seeds value
    Eigen::Vector3f seeds_mean = pc_mean.head<3>();

    // according to normal.T*[x,y,z] = -d
    d_ = -(normal_.transpose()*seeds_mean)(0,0);
    // set distance threhold to `th_dist - d`
    th_dist_d_ = th_dist_ - d_;
 
    // return the equation parameters
}


/*
    @brief Extract initial seeds of the given pointcloud sorted segment.
    This function filter ground seeds points accoring to heigt.
    This function will set the `g_ground_pc` to `g_seed_pc`.
    @param p_sorted: sorted pointcloud
    
    @param ::num_lpr_: num of LPR points
    @param ::th_seeds_: threshold distance of seeds
    @param ::

*/
void extract_initial_seeds_(const pcl::PointCloud<PointType>& p_sorted){
    // LPR is the mean of low point representative
    double sum = 0;
    int cnt = 0;
    // Calculate the mean height value.
    for(int i=0;i<p_sorted.points.size() && cnt<num_lpr_;i++){
        sum += p_sorted.points[i].z;
        cnt++;
    }
    double lpr_height = cnt!=0?sum/cnt:0;// in case divide by 0
    g_seeds_pc->clear();
    // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
    for(int i=0;i<p_sorted.points.size();i++){
        if(p_sorted.points[i].z < lpr_height + th_seeds_){
            g_seeds_pc->points.push_back(p_sorted.points[i]);
        }
    }
    // return seeds points
}



int main(int argc, char **argv)
{

    ros::init(argc, argv, "GroundPlaneFit");
    ros::NodeHandle node_handle_;

    // Init ROS related
    ROS_INFO("Inititalizing Ground Plane Fitter...");

    node_handle_.param<int>("sensor_model", sensor_model_, 32);
    ROS_INFO("Sensor Model: %d", sensor_model_);

    node_handle_.param<double>("sensor_height", sensor_height_, 2.5);
    ROS_INFO("Sensor Height: %f", sensor_height_);

    node_handle_.param<int>("num_seg", num_seg_, 1);
    ROS_INFO("Num of Segments: %d", num_seg_);

    node_handle_.param<int>("num_iter", num_iter_, 3);
    ROS_INFO("Num of Iteration: %d", num_iter_);

    node_handle_.param<int>("num_lpr", num_lpr_, 20);
    ROS_INFO("Num of LPR: %d", num_lpr_);

    node_handle_.param<double>("th_seeds", th_seeds_, 1.2);
    ROS_INFO("Seeds Threshold: %f", th_seeds_);

    node_handle_.param<double>("th_dist", th_dist_, 0.3);
    ROS_INFO("Distance Threshold: %f", th_dist_);

    node_handle_.param<string>("data_path", data_path, "/");
    ROS_INFO("data_path: %s", data_path);

    node_handle_.param<string>("pcd_savepath", pcd_savepath, "/");
    ROS_INFO("pcd_savepath: %s", pcd_savepath);

    node_handle_.param<bool>("save_csv", save_csv, false);
    node_handle_.param<bool>("save_flag", save_flag, false);
    ROS_INFO("save_flag: %d", save_flag);

    node_handle_.param<string>("output_gtpoints_name", output_gtpoints_name, "/");
    node_handle_.param<string>("output_filename", output_filename, "/");


    CloudPublisher     = node_handle_.advertise<sensor_msgs::PointCloud2>("/benchmark/cloud", 100, true);
    TPPublisher        = node_handle_.advertise<sensor_msgs::PointCloud2>("/benchmark/TP", 100, true);
    FPPublisher        = node_handle_.advertise<sensor_msgs::PointCloud2>("/benchmark/FP", 100, true);
    FNPublisher        = node_handle_.advertise<sensor_msgs::PointCloud2>("/benchmark/FN", 100, true);
    PrecisionPublisher = node_handle_.advertise<visualization_msgs::Marker>("/precision", 1, true);
    RecallPublisher    = node_handle_.advertise<visualization_msgs::Marker>("/recall", 1, true);

    KittiLoader loader(data_path);
    int      N = loader.size();
    for (int n = 0; n < N; ++n) {
        cout << n << "th node come" << endl;
        pcl::PointCloud<PointType> laserCloudIn;
        loader.get_cloud(n, laserCloudIn);
        pcl::PointCloud<PointType> pc_ground;
        pcl::PointCloud<PointType> pc_non_ground;

        static double time_taken;
        double start = ros::Time::now().toSec();
        //std::vector<int> indices;
        //pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn,indices);
        // 2.Sort on Z-axis value.
        sort(laserCloudIn.points.begin(),laserCloudIn.end(),point_cmp);
        // 3.Error point removal
        // As there are some error mirror reflection under the ground, 
        // here regardless point under 2* sensor_height
        // Sort point according to height, here uses z-axis in default
        pcl::PointCloud<PointType>::iterator it = laserCloudIn.points.begin();
        for(int i=0;i<laserCloudIn.points.size();i++){
            if(laserCloudIn.points[i].z < -1.5*sensor_height_){
                it++;
            }else{
                break;
            }
        }
        laserCloudIn.points.erase(laserCloudIn.points.begin(),it);
        // 4. Extract init ground seeds.
        extract_initial_seeds_(laserCloudIn);
        g_ground_pc = g_seeds_pc;

    
        // 5. Ground plane fitter mainloop
        for(int i=0;i<num_iter_;i++){
            estimate_plane_();
            g_ground_pc->clear();
            g_not_ground_pc->clear();

            //pointcloud to matrix
            MatrixXf points(laserCloudIn.points.size(),3);
            int j =0;
            for(auto p:laserCloudIn.points){
                points.row(j++)<<p.x,p.y,p.z;
            }
            // ground plane model
            VectorXf result = points*normal_;
            // threshold filter
            for(int r=0;r<result.rows();r++){
                if(result[r]<th_dist_d_){
                    g_ground_pc->points.push_back(laserCloudIn[r]);
                }else{
                    g_not_ground_pc->points.push_back(laserCloudIn[r]);
                }
            }
        }
        pc_ground = *g_ground_pc;
        pc_non_ground = *g_not_ground_pc;

        double end = ros::Time::now().toSec();
        time_taken = end -start;
        // Estimation
        double precision, recall, precision_naive, recall_naive;
        int gt_ground_nums;
        calculate_precision_recall(laserCloudIn, pc_ground, precision, recall, gt_ground_nums);
        calculate_precision_recall(laserCloudIn, pc_ground, precision_naive, recall_naive, gt_ground_nums, false);
        cout << "\033[1;33m" << n << "th, " << "gt ground nums: " << gt_ground_nums << "\033[0m" << endl;
        cout << "\033[1;32m" << n << "th, " << " takes : " << time_taken << " | " << laserCloudIn.size() << " -> "
             << pc_ground.size()
             << "\033[0m" << endl;

        cout << "\033[1;32m P: " << precision << " | R: " << recall << "\033[0m" << endl;


        // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        //        If you want to save precision/recall in a text file, revise this part
        // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        if(save_csv){
            ofstream ground_output(output_filename, ios::app);
            ground_output << n << "," << time_taken << "," << precision << "," << recall << "," << precision_naive << "," << recall_naive;
            ground_output << std::endl;
            ground_output.close();
        }


        // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

        // Publish msg
        pcl::PointCloud<PointType> TP;
        pcl::PointCloud<PointType> FP;
        pcl::PointCloud<PointType> FN;
        pcl::PointCloud<PointType> TN;
        discern_ground(pc_ground, TP, FP);
        discern_ground(pc_non_ground, FN, TN);

        // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        //        If you want to save the output of pcd, revise this part
        // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        if (save_flag) {
            std::map<int, int> pc_curr_gt_counts, g_est_gt_counts;

            std::string count_str        = std::to_string(n);
            std::string count_str_padded = std::string(NUM_ZEROS - count_str.length(), '0') + count_str;
            std::string pcd_filename     = pcd_savepath + "/" + count_str_padded + ".pcd";
            pc2pcdfile(TP, FP, FN, TN, pcd_filename);
        }
        // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        laserCloudIn.width = laserCloudIn.points.size();
        laserCloudIn.height = 1;
        CloudPublisher.publish(cloud2msg(laserCloudIn));
        TPPublisher.publish(cloud2msg(TP));
        FPPublisher.publish(cloud2msg(FP));
        FNPublisher.publish(cloud2msg(FN));
        pub_score("p", precision);
        pub_score("r", recall);
    }


    ros::spin();

    return 0;

}