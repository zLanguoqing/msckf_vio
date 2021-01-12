/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <iterator>
#include <algorithm>

#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/SparseCore>
#include <Eigen/SPQRSupport>
#include <boost/math/distributions/chi_squared.hpp>

#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <msckf_vio/msckf_vio.h>
#include <msckf_vio/math_utils.hpp>
#include <msckf_vio/utils.h>

using namespace std;
using namespace Eigen;

namespace msckf_vio{
// Static member variables in IMUState class.
StateIDType IMUState::next_id = 0;
double IMUState::gyro_noise = 0.001;
double IMUState::acc_noise = 0.01;
double IMUState::gyro_bias_noise = 0.001;
double IMUState::acc_bias_noise = 0.01;
Vector3d IMUState::gravity = Vector3d(0, 0, -GRAVITY_ACCELERATION);
Isometry3d IMUState::T_imu_body = Isometry3d::Identity();

// Static member variables in CAMState class.
Isometry3d CAMState::T_cam0_cam1 = Isometry3d::Identity();

// Static member variables in Feature class.
FeatureIDType Feature::next_id = 0;
double Feature::observation_noise = 0.01;
Feature::OptimizationConfig Feature::optimization_config;

map<int, double> MsckfVio::chi_squared_test_table;

MsckfVio::MsckfVio(ros::NodeHandle& pnh):
  is_gravity_set(false),
  is_first_img(true),
  nh(pnh) {
  return;
}
/*加载滤波所需参数*/
/**
 * @brief 加载滤波所需参数以及相对应的初始值
 * 
 * @return  bool
 */
bool MsckfVio::loadParameters() {
  // Frame id
  nh.param<string>("fixed_frame_id", fixed_frame_id, "world");
  nh.param<string>("child_frame_id", child_frame_id, "robot");
  nh.param<bool>("publish_tf", publish_tf, true);
  nh.param<double>("frame_rate", frame_rate, 40.0);
  nh.param<double>("position_std_threshold", position_std_threshold, 8.0);

  nh.param<double>("rotation_threshold", rotation_threshold, 0.2618);
  nh.param<double>("translation_threshold", translation_threshold, 0.4);
  nh.param<double>("tracking_rate_threshold", tracking_rate_threshold, 0.5);

  // Feature optimization parameters
  nh.param<double>("feature/config/translation_threshold",
      Feature::optimization_config.translation_threshold, 0.2);

  // Noise related parameters
  nh.param<double>("noise/gyro", IMUState::gyro_noise, 0.001);
  nh.param<double>("noise/acc", IMUState::acc_noise, 0.01);
  nh.param<double>("noise/gyro_bias", IMUState::gyro_bias_noise, 0.001);
  nh.param<double>("noise/acc_bias", IMUState::acc_bias_noise, 0.01);
  nh.param<double>("noise/feature", Feature::observation_noise, 0.01);

  // Use variance instead of standard deviation.
  IMUState::gyro_noise *= IMUState::gyro_noise;
  IMUState::acc_noise *= IMUState::acc_noise;
  IMUState::gyro_bias_noise *= IMUState::gyro_bias_noise;
  IMUState::acc_bias_noise *= IMUState::acc_bias_noise;
  Feature::observation_noise *= Feature::observation_noise;

  // Set the initial IMU state.
  // The intial orientation and position will be set to the origin
  // implicitly. But the initial velocity and bias can be
  // set by parameters.
  // TODO: is it reasonable to set the initial bias to 0?
  // IMU系统速度的初始值
  nh.param<double>("initial_state/velocity/x",
      state_server.imu_state.velocity(0), 0.0);
  nh.param<double>("initial_state/velocity/y",
      state_server.imu_state.velocity(1), 0.0);
  nh.param<double>("initial_state/velocity/z",
      state_server.imu_state.velocity(2), 0.0);

  // The initial covariance of orientation and position can be
  // set to 0. But for velocity, bias and extrinsic parameters,
  // there should be nontrivial uncertainty.
  // 加速度 线速度 速度的协方差初始值
  double gyro_bias_cov, acc_bias_cov, velocity_cov;
  nh.param<double>("initial_covariance/velocity",
      velocity_cov, 0.25);
  nh.param<double>("initial_covariance/gyro_bias",
      gyro_bias_cov, 1e-4);
  nh.param<double>("initial_covariance/acc_bias",
      acc_bias_cov, 1e-2);

  // 外参，相机和IMU的，旋转量的协方差和平移量的协方差初始值
  double extrinsic_rotation_cov, extrinsic_translation_cov;
  nh.param<double>("initial_covariance/extrinsic_rotation_cov",
      extrinsic_rotation_cov, 3.0462e-4);
  nh.param<double>("initial_covariance/extrinsic_translation_cov",
      extrinsic_translation_cov, 1e-4);
  // 系统协方差的初始值
  state_server.state_cov = MatrixXd::Zero(21, 21);
  for (int i = 3; i < 6; ++i)
    state_server.state_cov(i, i) = gyro_bias_cov;
  for (int i = 6; i < 9; ++i)
    state_server.state_cov(i, i) = velocity_cov;
  for (int i = 9; i < 12; ++i)
    state_server.state_cov(i, i) = acc_bias_cov;
  for (int i = 15; i < 18; ++i)
    state_server.state_cov(i, i) = extrinsic_rotation_cov;
  for (int i = 18; i < 21; ++i)
    state_server.state_cov(i, i) = extrinsic_translation_cov;

  // Transformation offsets between the frames involved.
  // 外参  T_imu_cam0 表示 IMU 2 cam
  Isometry3d T_imu_cam0 = utils::getTransformEigen(nh, "cam0/T_cam_imu");
  Isometry3d T_cam0_imu = T_imu_cam0.inverse();
  // 
  state_server.imu_state.R_imu_cam0 = T_cam0_imu.linear().transpose();
  state_server.imu_state.t_cam0_imu = T_cam0_imu.translation();
  CAMState::T_cam0_cam1 =
    utils::getTransformEigen(nh, "cam1/T_cn_cnm1");
  IMUState::T_imu_body =
    utils::getTransformEigen(nh, "T_imu_body").inverse();

  // Maximum number of camera states to be stored
  nh.param<int>("max_cam_state_size", max_cam_state_size, 30);

  ROS_INFO("===========================================");
  ROS_INFO("fixed frame id: %s", fixed_frame_id.c_str());
  ROS_INFO("child frame id: %s", child_frame_id.c_str());
  ROS_INFO("publish tf: %d", publish_tf);
  ROS_INFO("frame rate: %f", frame_rate);
  ROS_INFO("position std threshold: %f", position_std_threshold);
  ROS_INFO("Keyframe rotation threshold: %f", rotation_threshold);
  ROS_INFO("Keyframe translation threshold: %f", translation_threshold);
  ROS_INFO("Keyframe tracking rate threshold: %f", tracking_rate_threshold);
  ROS_INFO("gyro noise: %.10f", IMUState::gyro_noise);
  ROS_INFO("gyro bias noise: %.10f", IMUState::gyro_bias_noise);
  ROS_INFO("acc noise: %.10f", IMUState::acc_noise);
  ROS_INFO("acc bias noise: %.10f", IMUState::acc_bias_noise);
  ROS_INFO("observation noise: %.10f", Feature::observation_noise);
  ROS_INFO("initial velocity: %f, %f, %f",
      state_server.imu_state.velocity(0),
      state_server.imu_state.velocity(1),
      state_server.imu_state.velocity(2));
  ROS_INFO("initial gyro bias cov: %f", gyro_bias_cov);
  ROS_INFO("initial acc bias cov: %f", acc_bias_cov);
  ROS_INFO("initial velocity cov: %f", velocity_cov);
  ROS_INFO("initial extrinsic rotation cov: %f",
      extrinsic_rotation_cov);
  ROS_INFO("initial extrinsic translation cov: %f",
      extrinsic_translation_cov);

  cout << T_imu_cam0.linear() << endl;
  cout << T_imu_cam0.translation().transpose() << endl;

  ROS_INFO("max camera state #: %d", max_cam_state_size);
  ROS_INFO("===========================================");
  return true;
}

/**
 * @brief 开启ROS接受前端信息的接口
 * 
 * @return  bool
 */
bool MsckfVio::createRosIO() {
  // odom_pub 里程计发布主题
  odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 10);
  // feature_pub 特征点发布主题
  feature_pub = nh.advertise<sensor_msgs::PointCloud2>(
      "feature_point_cloud", 10);
  // 系统丢失后重新初始化系统 resetCallback为重出始化接口
  // 没有重定位功能
  reset_srv = nh.advertiseService("reset",
      &MsckfVio::resetCallback, this);
// imu接收数据接口imuCallback
  imu_sub = nh.subscribe("imu", 100,
      &MsckfVio::imuCallback, this);
// 特征点接收数据featureCallback
  feature_sub = nh.subscribe("features", 40,
      &MsckfVio::featureCallback, this);

  mocap_odom_sub = nh.subscribe("mocap_odom", 10,
      &MsckfVio::mocapOdomCallback, this);
  mocap_odom_pub = nh.advertise<nav_msgs::Odometry>("gt_odom", 1);

  return true;
}
/**
 * @brief msckf系统初始化，包括系统协方差，噪声方差，同时开启ROS接受前端数据
 * 
 * @return  bool
 */
bool MsckfVio::initialize() {
  // 加载msckf所需参数
  if (!loadParameters()) return false;
  ROS_INFO("Finish loading ROS parameters...");

  // Initialize state server
  // 系统噪声矩阵 bg ng ba na
  state_server.continuous_noise_cov =
    Matrix<double, 12, 12>::Zero();
  state_server.continuous_noise_cov.block<3, 3>(0, 0) =
    Matrix3d::Identity()*IMUState::gyro_noise;
  state_server.continuous_noise_cov.block<3, 3>(3, 3) =
    Matrix3d::Identity()*IMUState::gyro_bias_noise;
  state_server.continuous_noise_cov.block<3, 3>(6, 6) =
    Matrix3d::Identity()*IMUState::acc_noise;
  state_server.continuous_noise_cov.block<3, 3>(9, 9) =
    Matrix3d::Identity()*IMUState::acc_bias_noise;

  // Initialize the chi squared test table with confidence
  // level 0.95.
  // 卡方检验表格，用于检验特征点观测是否正确
  for (int i = 1; i < 100; ++i) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_test_table[i] =
      boost::math::quantile(chi_squared_dist, 0.05);
  }

  if (!createRosIO()) return false;
  ROS_INFO("Finish creating ROS IO...");

  return true;
}
/**
 * @brief msckf系统初始化，包括系统协方差，噪声方差，同时开启ROS接受前端数据
 * @param msg          // 前端IMU数据数据
 * @return  void
 */
void MsckfVio::imuCallback(
    const sensor_msgs::ImuConstPtr& msg) {

  // IMU msgs are pushed backed into a buffer instead of
  // being processed immediately. The IMU msgs are processed
  // when the next image is available, in which way, we can
  // easily handle the transfer delay.
  // 把IMU数据放入imu_msg_buffer中
  imu_msg_buffer.push_back(*msg);
  // is_gravity_set IMU是否初始化过，如果没有初始化过，则进入初始化
  if (!is_gravity_set) {
    // 需要充足的IMU数据进行初始化，如果没有200帧的IMU数据则不初始化
    if (imu_msg_buffer.size() < 200) return;
    //if (imu_msg_buffer.size() < 10) return;
    // 初始化，静态初始化就一定成功吗？
    initializeGravityAndBias();
    // is_gravity_set设置为true 下次就不在进入
    is_gravity_set = true;
  }

  return;
}
/**
 * @brief IMU静态初始化，得到bg和旋转量初始值

 * @return  void
 */
void MsckfVio::initializeGravityAndBias() {

  // Initialize gravity and gyro bias.
  Vector3d sum_angular_vel = Vector3d::Zero();
  Vector3d sum_linear_acc = Vector3d::Zero();
  // 把之前的200帧IMU数据计算角速度的平均值和加速度的平均值
  for (const auto& imu_msg : imu_msg_buffer) {
    Vector3d angular_vel = Vector3d::Zero();
    Vector3d linear_acc = Vector3d::Zero();
    // 数据类型的转换
    tf::vectorMsgToEigen(imu_msg.angular_velocity, angular_vel);
    tf::vectorMsgToEigen(imu_msg.linear_acceleration, linear_acc);

    sum_angular_vel += angular_vel;
    sum_linear_acc += linear_acc;
  }
  // 把角速度的平均值当做bg初始值
  state_server.imu_state.gyro_bias =
    sum_angular_vel / imu_msg_buffer.size();
  //IMUState::gravity =
  //  -sum_linear_acc / imu_msg_buffer.size();
  // This is the gravity in the IMU frame.
  // 计算加速度的平均值
  Vector3d gravity_imu =
    sum_linear_acc / imu_msg_buffer.size();

  // Initialize the initial orientation, so that the estimation
  // is consistent with the inertial frame.
  // 加速度的模长，用这个加速度的模长当做IMU重力模长
  double gravity_norm = gravity_imu.norm();

  IMUState::gravity = Vector3d(0.0, 0.0, -gravity_norm);
  // gravity_imu IMU的重力方向，代表真实的重力方向
  // IMUState::gravity 表示SLAM系统的重力方向
  // FromTwoVectors 计算gravity_imu和IMUState::gravity之间的夹角
  // q0_i_w 说明可以将SLAM坐标系与真实世界坐标系对齐
  Quaterniond q0_i_w = Quaterniond::FromTwoVectors(
    gravity_imu, -IMUState::gravity);
  state_server.imu_state.orientation =
    rotationToQuaternion(q0_i_w.toRotationMatrix().transpose());

  return;
}

bool MsckfVio::resetCallback(
    std_srvs::Trigger::Request& req,
    std_srvs::Trigger::Response& res) {

  ROS_WARN("Start resetting msckf vio...");
  // Temporarily shutdown the subscribers to prevent the
  // state from updating.
  feature_sub.shutdown();
  imu_sub.shutdown();

  // Reset the IMU state.
  IMUState& imu_state = state_server.imu_state;
  imu_state.time = 0.0;
  imu_state.orientation = Vector4d(0.0, 0.0, 0.0, 1.0);
  imu_state.position = Vector3d::Zero();
  imu_state.velocity = Vector3d::Zero();
  imu_state.gyro_bias = Vector3d::Zero();
  imu_state.acc_bias = Vector3d::Zero();
  imu_state.orientation_null = Vector4d(0.0, 0.0, 0.0, 1.0);
  imu_state.position_null = Vector3d::Zero();
  imu_state.velocity_null = Vector3d::Zero();

  // Remove all existing camera states.
  state_server.cam_states.clear();

  // Reset the state covariance.
  double gyro_bias_cov, acc_bias_cov, velocity_cov;
  nh.param<double>("initial_covariance/velocity",
      velocity_cov, 0.25);
  nh.param<double>("initial_covariance/gyro_bias",
      gyro_bias_cov, 1e-4);
  nh.param<double>("initial_covariance/acc_bias",
      acc_bias_cov, 1e-2);

  double extrinsic_rotation_cov, extrinsic_translation_cov;
  nh.param<double>("initial_covariance/extrinsic_rotation_cov",
      extrinsic_rotation_cov, 3.0462e-4);
  nh.param<double>("initial_covariance/extrinsic_translation_cov",
      extrinsic_translation_cov, 1e-4);

  state_server.state_cov = MatrixXd::Zero(21, 21);
  for (int i = 3; i < 6; ++i)
    state_server.state_cov(i, i) = gyro_bias_cov;
  for (int i = 6; i < 9; ++i)
    state_server.state_cov(i, i) = velocity_cov;
  for (int i = 9; i < 12; ++i)
    state_server.state_cov(i, i) = acc_bias_cov;
  for (int i = 15; i < 18; ++i)
    state_server.state_cov(i, i) = extrinsic_rotation_cov;
  for (int i = 18; i < 21; ++i)
    state_server.state_cov(i, i) = extrinsic_translation_cov;

  // Clear all exsiting features in the map.
  map_server.clear();

  // Clear the IMU msg buffer.
  imu_msg_buffer.clear();

  // Reset the starting flags.
  is_gravity_set = false;
  is_first_img = true;

  // Restart the subscribers.
  imu_sub = nh.subscribe("imu", 100,
      &MsckfVio::imuCallback, this);
  feature_sub = nh.subscribe("features", 40,
      &MsckfVio::featureCallback, this);

  // TODO: When can the reset fail?
  res.success = true;
  ROS_WARN("Resetting msckf vio completed...");
  return true;
}
/**
 * @brief msckf主函数，后端优化都是这个函数
 * @param msg          // 前端图像数据数据，特征点信息
 * @return  void
 */
void MsckfVio::featureCallback(
    const CameraMeasurementConstPtr& msg) {

  // Return if the gravity vector has not been set.
  // 是否初始化
  if (!is_gravity_set) return;

  // Start the system if the first image is received.
  // The frame where the first image is received will be
  // the origin.
  // 是否是第一帧图像信息
  // 记录第一帧图像的时间作为系统状态时间
  if (is_first_img) {
    is_first_img = false;
    state_server.imu_state.time = msg->header.stamp.toSec();
  }

  static double max_processing_time = 0.0;
  static int critical_time_cntr = 0;
  double processing_start_time = ros::Time::now().toSec();

  // Propogate the IMU state.
  // that are received before the image msg.
  ros::Time start_time = ros::Time::now();
  // imu预积分与系统协方差矩阵传播
  batchImuProcessing(msg->header.stamp.toSec());
  double imu_processing_time = (
      ros::Time::now()-start_time).toSec();

  // Augment the state vector.
  start_time = ros::Time::now();
  // 计算增广矩阵与相机的状态
  stateAugmentation(msg->header.stamp.toSec());
  double state_augmentation_time = (
      ros::Time::now()-start_time).toSec();

  // Add new observations for existing features or new
  // features in the map server.
  start_time = ros::Time::now();
  // 添加前端的特征点信息
  addFeatureObservations(msg);
  double add_observations_time = (
      ros::Time::now()-start_time).toSec();

  // Perform measurement update if necessary.
  start_time = ros::Time::now();
  // 利用跟踪丢失的特征点约束进行滤波
  // 跟踪未丢失则不进行滤波
  removeLostFeatures();
  double remove_lost_features_time = (
      ros::Time::now()-start_time).toSec();

  start_time = ros::Time::now();
  pruneCamStateBuffer();
  double prune_cam_states_time = (
      ros::Time::now()-start_time).toSec();

  // Publish the odometry.
  start_time = ros::Time::now();
  publish(msg->header.stamp);
  double publish_time = (
      ros::Time::now()-start_time).toSec();

  // Reset the system if necessary.
  onlineReset();

  double processing_end_time = ros::Time::now().toSec();
  double processing_time =
    processing_end_time - processing_start_time;
  if (processing_time > 1.0/frame_rate) {
    ++critical_time_cntr;
    ROS_INFO("\033[1;31mTotal processing time %f/%d...\033[0m",
        processing_time, critical_time_cntr);
    //printf("IMU processing time: %f/%f\n",
    //    imu_processing_time, imu_processing_time/processing_time);
    //printf("State augmentation time: %f/%f\n",
    //    state_augmentation_time, state_augmentation_time/processing_time);
    //printf("Add observations time: %f/%f\n",
    //    add_observations_time, add_observations_time/processing_time);
    printf("Remove lost features time: %f/%f\n",
        remove_lost_features_time, remove_lost_features_time/processing_time);
    printf("Remove camera states time: %f/%f\n",
        prune_cam_states_time, prune_cam_states_time/processing_time);
    //printf("Publish time: %f/%f\n",
    //    publish_time, publish_time/processing_time);
  }

  return;
}

void MsckfVio::mocapOdomCallback(
    const nav_msgs::OdometryConstPtr& msg) {
  static bool first_mocap_odom_msg = true;

  // If this is the first mocap odometry messsage, set
  // the initial frame.
  if (first_mocap_odom_msg) {
    Quaterniond orientation;
    Vector3d translation;
    tf::pointMsgToEigen(
        msg->pose.pose.position, translation);
    tf::quaternionMsgToEigen(
        msg->pose.pose.orientation, orientation);
    //tf::vectorMsgToEigen(
    //    msg->transform.translation, translation);
    //tf::quaternionMsgToEigen(
    //    msg->transform.rotation, orientation);
    mocap_initial_frame.linear() = orientation.toRotationMatrix();
    mocap_initial_frame.translation() = translation;
    first_mocap_odom_msg = false;
  }

  // Transform the ground truth.
  Quaterniond orientation;
  Vector3d translation;
  //tf::vectorMsgToEigen(
  //    msg->transform.translation, translation);
  //tf::quaternionMsgToEigen(
  //    msg->transform.rotation, orientation);
  tf::pointMsgToEigen(
      msg->pose.pose.position, translation);
  tf::quaternionMsgToEigen(
      msg->pose.pose.orientation, orientation);

  Eigen::Isometry3d T_b_v_gt;
  T_b_v_gt.linear() = orientation.toRotationMatrix();
  T_b_v_gt.translation() = translation;
  Eigen::Isometry3d T_b_w_gt = mocap_initial_frame.inverse() * T_b_v_gt;

  //Eigen::Vector3d body_velocity_gt;
  //tf::vectorMsgToEigen(msg->twist.twist.linear, body_velocity_gt);
  //body_velocity_gt = mocap_initial_frame.linear().transpose() *
  //  body_velocity_gt;

  // Ground truth tf.
  if (publish_tf) {
    tf::Transform T_b_w_gt_tf;
    tf::transformEigenToTF(T_b_w_gt, T_b_w_gt_tf);
    tf_pub.sendTransform(tf::StampedTransform(
          T_b_w_gt_tf, msg->header.stamp, fixed_frame_id, child_frame_id+"_mocap"));
  }

  // Ground truth odometry.
  nav_msgs::Odometry mocap_odom_msg;
  mocap_odom_msg.header.stamp = msg->header.stamp;
  mocap_odom_msg.header.frame_id = fixed_frame_id;
  mocap_odom_msg.child_frame_id = child_frame_id+"_mocap";

  tf::poseEigenToMsg(T_b_w_gt, mocap_odom_msg.pose.pose);
  //tf::vectorEigenToMsg(body_velocity_gt,
  //    mocap_odom_msg.twist.twist.linear);

  mocap_odom_pub.publish(mocap_odom_msg);
  return;
}
/**
 * @brief IMU数据处理，主要是预积分和系统协方差的初步计算
 * @param time_bound          // 图像信息时间戳
 * @return  void
 */
void MsckfVio::batchImuProcessing(const double& time_bound) {
  // Counter how many IMU msgs in the buffer are used.
  // used_imu_msg_cntr 记录IMU的使用量，可以将使用过的IMU数据从
  // imu_msg_buffer中剔除，不重复使用
  int used_imu_msg_cntr = 0;

  for (const auto& imu_msg : imu_msg_buffer) {
    // 根据图像时间来选取IMU数据
    // imu_time 为IMU时间戳
    // imu_state.time 为上一时刻的系统时间戳
    // 如果IMU数据的时间小于上一时刻的，就没必要用了
    double imu_time = imu_msg.header.stamp.toSec();
    if (imu_time < state_server.imu_state.time) {
      ++used_imu_msg_cntr;
      continue;
    }
    // 如果大于当前图像的时间戳，也不用，只用上一时刻到当前时刻之间的时间戳
    // 这里IMU数据的选取不太严格
    if (imu_time > time_bound) break;

    // Convert the msgs.
    Vector3d m_gyro, m_acc;
    // m_gyro m_acc 代表角速度和加速度
    tf::vectorMsgToEigen(imu_msg.angular_velocity, m_gyro);
    tf::vectorMsgToEigen(imu_msg.linear_acceleration, m_acc);

    // Execute process model.
    // IMU预积分和系统协方差矩阵的计算
    processModel(imu_time, m_gyro, m_acc);
    ++used_imu_msg_cntr;
  }

  // Set the state ID for the new IMU state.
  state_server.imu_state.id = IMUState::next_id++;

  // Remove all used IMU msgs.
  imu_msg_buffer.erase(imu_msg_buffer.begin(),
      imu_msg_buffer.begin()+used_imu_msg_cntr);

  return;
}

/**
 * @brief F矩阵与G矩阵计算
 * @brief 计算转移矩阵，与噪声协方差矩阵
 * @brief IMU预积分，采用RK4方法进行积分
 * @param time          // 当前帧的IMU时间戳
 * @param m_gyro          // 当前帧的IMU的角速度
 * @param m_acc          // 当前帧的IMU加速度
 * @return  void
 */
void MsckfVio::processModel(const double& time,
    const Vector3d& m_gyro,
    const Vector3d& m_acc) {

  // Remove the bias from the measured gyro and acceleration
  IMUState& imu_state = state_server.imu_state;
  //现将IMU中的加速度和角速度去偏置
  Vector3d gyro = m_gyro - imu_state.gyro_bias;
  Vector3d acc = m_acc - imu_state.acc_bias;
  // 计算时间差
  double dtime = time - imu_state.time;

  // Compute discrete transition and noise covariance matrix
  // 论文中离散条件下的误差模型中的F矩阵和G矩阵
  Matrix<double, 21, 21> F = Matrix<double, 21, 21>::Zero();
  Matrix<double, 21, 12> G = Matrix<double, 21, 12>::Zero();
  // 根据公式得出
  F.block<3, 3>(0, 0) = -skewSymmetric(gyro);
  F.block<3, 3>(0, 3) = -Matrix3d::Identity();
  F.block<3, 3>(6, 0) = -quaternionToRotation(
      imu_state.orientation).transpose()*skewSymmetric(acc);
  F.block<3, 3>(6, 9) = -quaternionToRotation(
      imu_state.orientation).transpose();
  F.block<3, 3>(12, 6) = Matrix3d::Identity();

  G.block<3, 3>(0, 0) = -Matrix3d::Identity();
  G.block<3, 3>(3, 3) = Matrix3d::Identity();
  G.block<3, 3>(6, 6) = -quaternionToRotation(
      imu_state.orientation).transpose();
  G.block<3, 3>(9, 9) = Matrix3d::Identity();

  // Approximate matrix exponential to the 3rd order,
  // which can be considered to be accurate enough assuming
  // dtime is within 0.01s.
  //计算状态转移矩阵
  Matrix<double, 21, 21> Fdt = F * dtime;
  Matrix<double, 21, 21> Fdt_square = Fdt * Fdt;
  Matrix<double, 21, 21> Fdt_cube = Fdt_square * Fdt;
  Matrix<double, 21, 21> Phi = Matrix<double, 21, 21>::Identity() +
    Fdt + 0.5*Fdt_square + (1.0/6.0)*Fdt_cube;

  // Propogate the state using 4th order Runge-Kutta
  // 采用RK4进行预积分，并得到对当前状态的预测
  predictNewState(dtime, gyro, acc);

  // Modify the transition matrix
  // 由于在不同点线性化，导致零空间崩塌，所以需要维持零空间维数不变
  // 与黄老师提出的FEJ不同的是，该方法采用零空间按照传播公式传播，则零空间不会崩塌
  // 所以需要调整转移矩阵，但是其实零空间内本也会带来漂移，这一块他忽略了
  Matrix3d R_kk_1 = quaternionToRotation(imu_state.orientation_null);
  Phi.block<3, 3>(0, 0) =
    quaternionToRotation(imu_state.orientation) * R_kk_1.transpose();

  Vector3d u = R_kk_1 * IMUState::gravity;
  RowVector3d s = (u.transpose()*u).inverse() * u.transpose();

  Matrix3d A1 = Phi.block<3, 3>(6, 0);
  Vector3d w1 = skewSymmetric(
      imu_state.velocity_null-imu_state.velocity) * IMUState::gravity;
  Phi.block<3, 3>(6, 0) = A1 - (A1*u-w1)*s;

  Matrix3d A2 = Phi.block<3, 3>(12, 0);
  Vector3d w2 = skewSymmetric(
      dtime*imu_state.velocity_null+imu_state.position_null-
      imu_state.position) * IMUState::gravity;
  Phi.block<3, 3>(12, 0) = A2 - (A2*u-w2)*s;

  // Propogate the state covariance matrix.
  Matrix<double, 21, 21> Q = Phi*G*state_server.continuous_noise_cov*
    G.transpose()*Phi.transpose()*dtime;
    // 系统协方差矩阵传播
  state_server.state_cov.block<21, 21>(0, 0) =
    Phi*state_server.state_cov.block<21, 21>(0, 0)*Phi.transpose() + Q;
//如果存在相机，则需要传播相机与IMU之间的协方差
  if (state_server.cam_states.size() > 0) {
    state_server.state_cov.block(
        0, 21, 21, state_server.state_cov.cols()-21) =
      Phi * state_server.state_cov.block(
        0, 21, 21, state_server.state_cov.cols()-21);
    state_server.state_cov.block(
        21, 0, state_server.state_cov.rows()-21, 21) =
      state_server.state_cov.block(
        21, 0, state_server.state_cov.rows()-21, 21) * Phi.transpose();
  }
// 对称正定阵，数值稳定性好
  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  // Update the state correspondes to null space.
  // 维持零空间所需数据
  // IMU预测值
  imu_state.orientation_null = imu_state.orientation;
  imu_state.position_null = imu_state.position;
  imu_state.velocity_null = imu_state.velocity;

  // Update the state info
  state_server.imu_state.time = time;
  return;
}

/**
 * @brief IMU预积分
 * @param dt            // 时间差
 * @param gyro          // 当前帧的IMU的角速度
 * @param acc          // 当前帧的IMU加速度
 * @return  void
 */
void MsckfVio::predictNewState(const double& dt,
    const Vector3d& gyro,
    const Vector3d& acc) {

  // TODO: Will performing the forward integration using
  //    the inverse of the quaternion give better accuracy?
  double gyro_norm = gyro.norm();
  Matrix4d Omega = Matrix4d::Zero();
  Omega.block<3, 3>(0, 0) = -skewSymmetric(gyro);
  Omega.block<3, 1>(0, 3) = gyro;
  Omega.block<1, 3>(3, 0) = -gyro;
  // 先取出当前状态的位姿，这里是引用
  Vector4d& q = state_server.imu_state.orientation;
  Vector3d& v = state_server.imu_state.velocity;
  Vector3d& p = state_server.imu_state.position;

  // Some pre-calculation
  // 零阶四元数展开计算旋转量
  Vector4d dq_dt, dq_dt2;
  if (gyro_norm > 1e-5) {
    dq_dt = (cos(gyro_norm*dt*0.5)*Matrix4d::Identity() +
      1/gyro_norm*sin(gyro_norm*dt*0.5)*Omega) * q;
    dq_dt2 = (cos(gyro_norm*dt*0.25)*Matrix4d::Identity() +
      1/gyro_norm*sin(gyro_norm*dt*0.25)*Omega) * q;
  }
  else {
    dq_dt = (Matrix4d::Identity()+0.5*dt*Omega) *
      cos(gyro_norm*dt*0.5) * q;
    dq_dt2 = (Matrix4d::Identity()+0.25*dt*Omega) *
      cos(gyro_norm*dt*0.25) * q;
  }
  Matrix3d dR_dt_transpose = quaternionToRotation(dq_dt).transpose();
  Matrix3d dR_dt2_transpose = quaternionToRotation(dq_dt2).transpose();

  // k1 = f(tn, yn)
  Vector3d k1_v_dot = quaternionToRotation(q).transpose()*acc +
    IMUState::gravity;
  Vector3d k1_p_dot = v;

  // k2 = f(tn+dt/2, yn+k1*dt/2)
  Vector3d k1_v = v + k1_v_dot*dt/2;
  Vector3d k2_v_dot = dR_dt2_transpose*acc +
    IMUState::gravity;
  Vector3d k2_p_dot = k1_v;

  // k3 = f(tn+dt/2, yn+k2*dt/2)
  Vector3d k2_v = v + k2_v_dot*dt/2;
  Vector3d k3_v_dot = dR_dt2_transpose*acc +
    IMUState::gravity;
  Vector3d k3_p_dot = k2_v;

  // k4 = f(tn+dt, yn+k3*dt)
  Vector3d k3_v = v + k3_v_dot*dt;
  Vector3d k4_v_dot = dR_dt_transpose*acc +
    IMUState::gravity;
  Vector3d k4_p_dot = k3_v;

  // yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
  // 直接将预积分量加到当前状态上，得到对当前状态的预测值
  q = dq_dt;
  quaternionNormalize(q);
  v = v + dt/6*(k1_v_dot+2*k2_v_dot+2*k3_v_dot+k4_v_dot);
  p = p + dt/6*(k1_p_dot+2*k2_p_dot+2*k3_p_dot+k4_p_dot);

  return;
}
/**
 * @brief 来了新的相机信息之后，对系统协方差矩阵进行扩增
 * @param time            // 当前状态的信息

 * @return  void
 */
void MsckfVio::stateAugmentation(const double& time) {

  // IMU和相机的相对位姿，一般为左目
  const Matrix3d& R_i_c = state_server.imu_state.R_imu_cam0;
  const Vector3d& t_c_i = state_server.imu_state.t_cam0_imu;

  // Add a new camera state to the state server.
  // R_w_c 世界坐标系相对于相机坐标系的旋转量
  // t_c_w 相机相对于世界坐标系的平移量
  Matrix3d R_w_i = quaternionToRotation(
      state_server.imu_state.orientation);
  Matrix3d R_w_c = R_i_c * R_w_i;
  Vector3d t_c_w = state_server.imu_state.position +
    R_w_i.transpose()*t_c_i;
  // cam_states 保留的是当前窗口图像信息，后面滑窗的时候要用
  // id当前时刻的ID，将相机状态赋值ID
  // cam_state 接下去求相机的位姿，利用这个位姿可以求解观测方程
  state_server.cam_states[state_server.imu_state.id] =
    CAMState(state_server.imu_state.id);
  CAMState& cam_state = state_server.cam_states[
    state_server.imu_state.id];
  // 赋予相机状态时间
  // 通过之前的求解的预测值和IMU与相机的相对位姿，求解的相机的位姿
  cam_state.time = time;
  cam_state.orientation = rotationToQuaternion(R_w_c);
  cam_state.position = t_c_w;
  // 此处需要记录系统状态的预测值相对应的相机状态的预测值，用于后面的维持零空间不崩塌
  cam_state.orientation_null = cam_state.orientation;
  cam_state.position_null = cam_state.position;

  // Update the covariance matrix of the state.
  // To simplify computation, the matrix J below is the nontrivial block
  // in Equation (16) in "A Multi-State Constraint Kalman Filter for Vision
  // -aided Inertial Navigation".
  // 相机位姿对状态向量雅克比求解，用于协方差矩阵的增广
  // 先求 雅克比矩阵J
  Matrix<double, 6, 21> J = Matrix<double, 6, 21>::Zero();
  J.block<3, 3>(0, 0) = R_i_c;
  J.block<3, 3>(0, 15) = Matrix3d::Identity();
  J.block<3, 3>(3, 0) = skewSymmetric(R_w_i.transpose()*t_c_i);
  //J.block<3, 3>(3, 0) = -R_w_i.transpose()*skewSymmetric(t_c_i);
  J.block<3, 3>(3, 12) = Matrix3d::Identity();
  J.block<3, 3>(3, 18) = Matrix3d::Identity();

  // Resize the state covariance matrix.
  size_t old_rows = state_server.state_cov.rows();
  size_t old_cols = state_server.state_cov.cols();
  state_server.state_cov.conservativeResize(old_rows+6, old_cols+6);

  // Rename some matrix blocks for convenience.
  const Matrix<double, 21, 21>& P11 =
    state_server.state_cov.block<21, 21>(0, 0);
  const MatrixXd& P12 =
    state_server.state_cov.block(0, 21, 21, old_cols-21);

  // Fill in the augmented state covariance.
  // 增广矩阵的左下部分
  state_server.state_cov.block(old_rows, 0, 6, old_cols) << J*P11, J*P12;
  // 通过转置，计算矩阵的右上部分
  state_server.state_cov.block(0, old_cols, old_rows, 6) =
    state_server.state_cov.block(old_rows, 0, 6, old_cols).transpose();
  // 协方差矩阵的右下部分
  state_server.state_cov.block<6, 6>(old_rows, old_cols) =
    J * P11 * J.transpose();

  // Fix the covariance to be symmetric
  // 
  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  return;
}
/**
 * @brief 通过前端传入的特征信息进行稀疏地图的构建
 * @param msg            // 左右目特征信息

 * @return  void
 */
void MsckfVio::addFeatureObservations(
    const CameraMeasurementConstPtr& msg) {
  // 取出当前帧的ID
  StateIDType state_id = state_server.imu_state.id;
  // 当前时刻map_server中特征点个数
  int curr_feature_num = map_server.size();
  int tracked_feature_num = 0;

  // Add new observations for existing features or new
  // features in the map server.
  //将特征点信息按照ID放入map_server结构中
  for (const auto& feature : msg->features) {
    // 如果在地图中没找到对应的ID，那么认为是新的特征点
    if (map_server.find(feature.id) == map_server.end()) {
      // This is a new feature.
      map_server[feature.id] = Feature(feature.id);
      // 将对应的双目观测信息放入observations中
      map_server[feature.id].observations[state_id] =
        Vector4d(feature.u0, feature.v0,
            feature.u1, feature.v1);
    } else {
      // This is an old feature.
      // 如果找到了就相同的特征点ID下放入观测信息
      map_server[feature.id].observations[state_id] =
        Vector4d(feature.u0, feature.v0,
            feature.u1, feature.v1);
      // 计算跟踪率
      ++tracked_feature_num;
    }
  }
  //跟踪率
  tracking_rate =
    static_cast<double>(tracked_feature_num) /
    static_cast<double>(curr_feature_num);

  return;
}
/**
 * @brief 计算一个特征点中的一个观测所对应的雅克比矩阵
 * @param cam_state_id            // 相机ID
 * @param  feature_id             // 特征点id
 * @param  H_x                    // 状态量对应的雅克比矩阵
 * @param  H_f                    // 特征点对应的雅克比矩阵
 * @param  r                      // 残差
 * @return  void
 */
void MsckfVio::measurementJacobian(
    const StateIDType& cam_state_id,
    const FeatureIDType& feature_id,
    Matrix<double, 4, 6>& H_x, Matrix<double, 4, 3>& H_f, Vector4d& r) {
  // 现将特征点信息和相机状态信息取出
  // Prepare all the required data.
  const CAMState& cam_state = state_server.cam_states[cam_state_id];
  const Feature& feature = map_server[feature_id];

  // Cam0 pose.
  Matrix3d R_w_c0 = quaternionToRotation(cam_state.orientation);
  const Vector3d& t_c0_w = cam_state.position;

  // Cam1 pose.
  Matrix3d R_c0_c1 = CAMState::T_cam0_cam1.linear();
  Matrix3d R_w_c1 = CAMState::T_cam0_cam1.linear() * R_w_c0;
  Vector3d t_c1_w = t_c0_w - R_w_c1.transpose()*CAMState::T_cam0_cam1.translation();

  // 3d feature position in the world frame.
  // And its observation with the stereo cameras.
  const Vector3d& p_w = feature.position;
  const Vector4d& z = feature.observations.find(cam_state_id)->second;

  // Convert the feature position from the world frame to
  // the cam0 and cam1 frame.
  Vector3d p_c0 = R_w_c0 * (p_w-t_c0_w);
  Vector3d p_c1 = R_w_c1 * (p_w-t_c1_w);

  // Compute the Jacobians.
  //计算雅克比，与论文公式相同
  Matrix<double, 4, 3> dz_dpc0 = Matrix<double, 4, 3>::Zero();
  dz_dpc0(0, 0) = 1 / p_c0(2);
  dz_dpc0(1, 1) = 1 / p_c0(2);
  dz_dpc0(0, 2) = -p_c0(0) / (p_c0(2)*p_c0(2));
  dz_dpc0(1, 2) = -p_c0(1) / (p_c0(2)*p_c0(2));

  Matrix<double, 4, 3> dz_dpc1 = Matrix<double, 4, 3>::Zero();
  dz_dpc1(2, 0) = 1 / p_c1(2);
  dz_dpc1(3, 1) = 1 / p_c1(2);
  dz_dpc1(2, 2) = -p_c1(0) / (p_c1(2)*p_c1(2));
  dz_dpc1(3, 2) = -p_c1(1) / (p_c1(2)*p_c1(2));

  Matrix<double, 3, 6> dpc0_dxc = Matrix<double, 3, 6>::Zero();
  dpc0_dxc.leftCols(3) = skewSymmetric(p_c0);
  dpc0_dxc.rightCols(3) = -R_w_c0;

  Matrix<double, 3, 6> dpc1_dxc = Matrix<double, 3, 6>::Zero();
  dpc1_dxc.leftCols(3) = R_c0_c1 * skewSymmetric(p_c0);
  dpc1_dxc.rightCols(3) = -R_w_c1;

  Matrix3d dpc0_dpg = R_w_c0;
  Matrix3d dpc1_dpg = R_w_c1;

  H_x = dz_dpc0*dpc0_dxc + dz_dpc1*dpc1_dxc;
  H_f = dz_dpc0*dpc0_dpg + dz_dpc1*dpc1_dpg;

  // Modifty the measurement Jacobian to ensure
  // observability constrain.
  // 为了维持零空间不崩塌所需的调整
  Matrix<double, 4, 6> A = H_x;
  Matrix<double, 6, 1> u = Matrix<double, 6, 1>::Zero();
  u.block<3, 1>(0, 0) = quaternionToRotation(
      cam_state.orientation_null) * IMUState::gravity;
  u.block<3, 1>(3, 0) = skewSymmetric(
      p_w-cam_state.position_null) * IMUState::gravity;
  H_x = A - A*u*(u.transpose()*u).inverse()*u.transpose();
  H_f = -H_x.block<4, 3>(0, 3);

  // Compute the residual.
  // 计算残差
  r = z - Vector4d(p_c0(0)/p_c0(2), p_c0(1)/p_c0(2),
      p_c1(0)/p_c1(2), p_c1(1)/p_c1(2));

  return;
}

/**
 * @brief 计算一个特征点多帧观测的雅克比矩阵
 * @param feature_id    特征点ID
 * @param cam_state_ids 特征点feature_id对应的相机状态
 * @param H_x           雅克比矩阵
 * @param r             残差
 * @return  void
 */
void MsckfVio::featureJacobian(
    const FeatureIDType& feature_id,
    const std::vector<StateIDType>& cam_state_ids,
    MatrixXd& H_x, VectorXd& r) {
    // 先将feature_id对应的信息取出
  const auto& feature = map_server[feature_id];

  // Check how many camera states in the provided camera
  // id camera has actually seen this feature.
  vector<StateIDType> valid_cam_state_ids(0);、
  // 再确认观测对应的相机是否存在
  for (const auto& cam_id : cam_state_ids) {
    if (feature.observations.find(cam_id) ==
        feature.observations.end()) continue;

    valid_cam_state_ids.push_back(cam_id);
  }

  int jacobian_row_size = 0;
  // 一个观测对应的4维，所有观测对应的是4 * valid_cam_state_ids.size()
  jacobian_row_size = 4 * valid_cam_state_ids.size();
  // 构建H_xj H_fj r_j
  MatrixXd H_xj = MatrixXd::Zero(jacobian_row_size,
      21+state_server.cam_states.size()*6);
  MatrixXd H_fj = MatrixXd::Zero(jacobian_row_size, 3);
  VectorXd r_j = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;
  // 遍历伊特特征点的所有观测
  for (const auto& cam_id : valid_cam_state_ids) {

    Matrix<double, 4, 6> H_xi = Matrix<double, 4, 6>::Zero();
    Matrix<double, 4, 3> H_fi = Matrix<double, 4, 3>::Zero();
    Vector4d r_i = Vector4d::Zero();
    // 计算一个特征点对一个观测的雅克比
    measurementJacobian(cam_id, feature.id, H_xi, H_fi, r_i);
    // 再将每个雅克比按块放入H_xj中，得到一个特征点多帧观测的雅克比
    auto cam_state_iter = state_server.cam_states.find(cam_id);
    int cam_state_cntr = std::distance(
        state_server.cam_states.begin(), cam_state_iter);

    // Stack the Jacobians.
    H_xj.block<4, 6>(stack_cntr, 21+6*cam_state_cntr) = H_xi;
    H_fj.block<4, 3>(stack_cntr, 0) = H_fi;
    r_j.segment<4>(stack_cntr) = r_i;
    stack_cntr += 4;
  }

  // Project the residual and Jacobians onto the nullspace
  // of H_fj.
  // 计算做零空间，消去特征点H_fj的影响
  JacobiSVD<MatrixXd> svd_helper(H_fj, ComputeFullU | ComputeThinV);
  MatrixXd A = svd_helper.matrixU().rightCols(
      jacobian_row_size - 3);

  H_x = A.transpose() * H_xj;
  r = A.transpose() * r_j;

  return;
}

void MsckfVio::measurementUpdate(
    const MatrixXd& H, const VectorXd& r) {

  if (H.rows() == 0 || r.rows() == 0) return;

  // Decompose the final Jacobian matrix to reduce computational
  // complexity as in Equation (28), (29).
  MatrixXd H_thin;
  VectorXd r_thin;

  if (H.rows() > H.cols()) {
    // Convert H to a sparse matrix.
    SparseMatrix<double> H_sparse = H.sparseView();

    // Perform QR decompostion on H_sparse.
    // 通过QR分解降低维度，提高计算效率
    SPQR<SparseMatrix<double> > spqr_helper;
    spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
    spqr_helper.compute(H_sparse);

    MatrixXd H_temp;
    VectorXd r_temp;
    (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
    (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

    H_thin = H_temp.topRows(21+state_server.cam_states.size()*6);
    r_thin = r_temp.head(21+state_server.cam_states.size()*6);

    //HouseholderQR<MatrixXd> qr_helper(H);
    //MatrixXd Q = qr_helper.householderQ();
    //MatrixXd Q1 = Q.leftCols(21+state_server.cam_states.size()*6);

    //H_thin = Q1.transpose() * H;
    //r_thin = Q1.transpose() * r;
  } else {
    H_thin = H;
    r_thin = r;
  }

  // Compute the Kalman gain.
  const MatrixXd& P = state_server.state_cov;
  MatrixXd S = H_thin*P*H_thin.transpose() +
      Feature::observation_noise*MatrixXd::Identity(
        H_thin.rows(), H_thin.rows());
  //MatrixXd K_transpose = S.fullPivHouseholderQr().solve(H_thin*P);
  MatrixXd K_transpose = S.ldlt().solve(H_thin*P);
  // 计算增益
  MatrixXd K = K_transpose.transpose();

  // Compute the error of the state.
  VectorXd delta_x = K * r_thin;

  // Update the IMU state.
  const VectorXd& delta_x_imu = delta_x.head<21>();

  if (//delta_x_imu.segment<3>(0).norm() > 0.15 ||
      //delta_x_imu.segment<3>(3).norm() > 0.15 ||
      delta_x_imu.segment<3>(6).norm() > 0.5 ||
      //delta_x_imu.segment<3>(9).norm() > 0.5 ||
      delta_x_imu.segment<3>(12).norm() > 1.0) {
    printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
    printf("delta position: %f\n", delta_x_imu.segment<3>(12).norm());
    ROS_WARN("Update change is too large.");
    //return;
  }

  const Vector4d dq_imu =
    smallAngleQuaternion(delta_x_imu.head<3>());
  state_server.imu_state.orientation = quaternionMultiplication(
      dq_imu, state_server.imu_state.orientation);
  state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(3);
  state_server.imu_state.velocity += delta_x_imu.segment<3>(6);
  state_server.imu_state.acc_bias += delta_x_imu.segment<3>(9);
  state_server.imu_state.position += delta_x_imu.segment<3>(12);

  const Vector4d dq_extrinsic =
    smallAngleQuaternion(delta_x_imu.segment<3>(15));
  state_server.imu_state.R_imu_cam0 = quaternionToRotation(
      dq_extrinsic) * state_server.imu_state.R_imu_cam0;
  state_server.imu_state.t_cam0_imu += delta_x_imu.segment<3>(18);

  // Update the camera states.
  auto cam_state_iter = state_server.cam_states.begin();
  for (int i = 0; i < state_server.cam_states.size();
      ++i, ++cam_state_iter) {
    const VectorXd& delta_x_cam = delta_x.segment<6>(21+i*6);
    const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
    cam_state_iter->second.orientation = quaternionMultiplication(
        dq_cam, cam_state_iter->second.orientation);
    cam_state_iter->second.position += delta_x_cam.tail<3>();
  }

  // Update state covariance.
  // 三种计算方式，正定矩阵数值更稳定
  MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K*H_thin;
  //state_server.state_cov = I_KH*state_server.state_cov*I_KH.transpose() +
  //  K*K.transpose()*Feature::observation_noise;
  state_server.state_cov = I_KH*state_server.state_cov;

  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  return;
}
/**
 * @brief 卡方检验，检测一个特征点对应的多帧观测能不能满足观测的规律
 * @param H   雅克比矩阵
 * @param r   残差
 * @param dof 卡方表需要的自由度
 * @return  bool
 */
bool MsckfVio::gatingTest(
    const MatrixXd& H, const VectorXd& r, const int& dof) {
  // 计算卡方值
  MatrixXd P1 = H * state_server.state_cov * H.transpose();
  MatrixXd P2 = Feature::observation_noise *
    MatrixXd::Identity(H.rows(), H.rows());
  double gamma = r.transpose() * (P1+P2).ldlt().solve(r);

  //cout << dof << " " << gamma << " " <<
  //  chi_squared_test_table[dof] << " ";
  // 通过自由度查表阈值
  if (gamma < chi_squared_test_table[dof]) {
    //cout << "passed" << endl;
    return true;
  } else {
    //cout << "failed" << endl;
    return false;
  }
}

/**
 * @brief 边缘化丢失的3D点进行ekf更新，得到系统状态估计值
 * @param 
 * @return  void
 */
void MsckfVio::removeLostFeatures() {

  // Remove the features that lost track.
  // BTW, find the size the final Jacobian matrix and residual vector.
  // jacobian_row_size 记录H矩阵的维度
  int jacobian_row_size = 0;
  // 不可用的特征点ID，后面需要从地图中清除
  vector<FeatureIDType> invalid_feature_ids(0);
  // 跟踪丢失的特征点ID，需要拿来通过观测方程进行滤波更新的
  vector<FeatureIDType> processed_feature_ids(0);
  //  先从map_server中遍历所有的特征点
  for (auto iter = map_server.begin();
      iter != map_server.end(); ++iter) {
    // Rename the feature to be checked.
    auto& feature = iter->second;

    // Pass the features that are still being tracked.
    // 观测次数必须大于3个，增加稳定性，
    // 在为了消去观测模型中特征点的影响，采用左零空间消掉Hf时观测需要大于2，否则将出现同时将HX消去的情况
    if (feature.observations.find(state_server.imu_state.id) !=
        feature.observations.end()) continue;
    if (feature.observations.size() < 3) {
      invalid_feature_ids.push_back(feature.id);
      continue;
    }

    // Check if the feature can be initialized if it
    // has not been.
    // 是否计算出世界坐标系的3D点，如果不能也将放入其中invalid_feature_ids进行剔除
    if (!feature.is_initialized) {
      // 判断是否发生了足够的移动，能够三角化成功
      if (!feature.checkMotion(state_server.cam_states)) {
        invalid_feature_ids.push_back(feature.id);
        continue;
      } else {
        // 三角化，计算3D点
        if(!feature.initializePosition(state_server.cam_states)) {
          invalid_feature_ids.push_back(feature.id);
          continue;
        }
      }
    }
    // 一个观测，由于存在左右目坐标，所以是4维，一个特征点可以被多帧观测到
    // 所以4×observations.size()，因为在求雅克比的时候需要剔除hf的影响，需要乘以左零空间
    // 所以需要减去3
    jacobian_row_size += 4*feature.observations.size() - 3;
    processed_feature_ids.push_back(feature.id);
  }

  //cout << "invalid/processed feature #: " <<
  //  invalid_feature_ids.size() << "/" <<
  //  processed_feature_ids.size() << endl;
  //cout << "jacobian row #: " << jacobian_row_size << endl;

  // Remove the features that do not have enough measurements.
  // 清除不可用的观测值
  for (const auto& feature_id : invalid_feature_ids)
    map_server.erase(feature_id);

  // Return if there is no lost feature to be processed.
  // 如果用于可更新的观测值为0，则返回
  if (processed_feature_ids.size() == 0) return;
  // 构建大的雅克比矩阵和残差
  MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
      21+6*state_server.cam_states.size());
  VectorXd r = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  // Process the features which lose track.
  // 遍历跟踪丢失的且观测大于3的特征点
  for (const auto& feature_id : processed_feature_ids) {
    // 先取出这个特征点的所有信息
    auto& feature = map_server[feature_id];

    vector<StateIDType> cam_state_ids(0);
    // 再将每个观测信息中对应的相机ID记录
    for (const auto& measurement : feature.observations)
      cam_state_ids.push_back(measurement.first);

    MatrixXd H_xj;
    VectorXd r_j;
    // 计算一个特征点对多帧观测的约束的雅克比与残差
    featureJacobian(feature.id, cam_state_ids, H_xj, r_j);
   // 卡方检验，通过则加入大的雅克比矩阵块中，进行滤波更新
    if (gatingTest(H_xj, r_j, cam_state_ids.size()-1)) {
      H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
      r.segment(stack_cntr, r_j.rows()) = r_j;
      stack_cntr += H_xj.rows();
    }

    // Put an upper bound on the row size of measurement Jacobian,
    // which helps guarantee the executation time.
    if (stack_cntr > 1500) break;
  }
  // 重新构建雅克比矩阵和残差
  H_x.conservativeResize(stack_cntr, H_x.cols());
  r.conservativeResize(stack_cntr);

  // Perform the measurement update step.
  // 滤波更新，计算增益
  measurementUpdate(H_x, r);

  // Remove all processed features from the map.
  // 边缘化
  for (const auto& feature_id : processed_feature_ids)
    map_server.erase(feature_id);

  return;
}

/**
 * @brief   滑窗，如果满足条件则剔除17帧，否则剔除第一帧
 * @param   rm_cam_state_ids    //需要剔除的相机ID
 * @return  void
 */
void MsckfVio::findRedundantCamStates(
    vector<StateIDType>& rm_cam_state_ids) {

  // Move the iterator to the key position.
  auto key_cam_state_iter = state_server.cam_states.end();
  for (int i = 0; i < 4; ++i)
    --key_cam_state_iter;//第16帧
  auto cam_state_iter = key_cam_state_iter;
  ++cam_state_iter;//第17帧
  auto first_cam_state_iter = state_server.cam_states.begin();

  // Pose of the key camera state.
  const Vector3d key_position =
    key_cam_state_iter->second.position;
  const Matrix3d key_rotation = quaternionToRotation(
      key_cam_state_iter->second.orientation);

  // Mark the camera states to be removed based on the
  // motion between states.
  //若第17帧和第16帧相差小于阈值
  //剔除第17帧，否则剔除第一帧
  for (int i = 0; i < 2; ++i) {
    const Vector3d position =
      cam_state_iter->second.position;
    const Matrix3d rotation = quaternionToRotation(
        cam_state_iter->second.orientation);

    double distance = (position-key_position).norm();
    double angle = AngleAxisd(
        rotation*key_rotation.transpose()).angle();

    if (angle < rotation_threshold &&
        distance < translation_threshold &&
        tracking_rate > tracking_rate_threshold) {
      rm_cam_state_ids.push_back(cam_state_iter->first);
      ++cam_state_iter;
    } else {
      rm_cam_state_ids.push_back(first_cam_state_iter->first);
      ++first_cam_state_iter;
    }
  }

  // Sort the elements in the output vector.
  sort(rm_cam_state_ids.begin(), rm_cam_state_ids.end());

  return;
}
 
/**
 * @brief   滑窗，并将剔除的帧上的特征点进行滤波更新
 * @param 
 * @return  void
 */
void MsckfVio::pruneCamStateBuffer() {
  // 如果没满，则不滑窗
  if (state_server.cam_states.size() < max_cam_state_size)
    return;

  // Find two camera states to be removed.
  vector<StateIDType> rm_cam_state_ids(0);
  // 计算需要剔除的相机帧
  findRedundantCamStates(rm_cam_state_ids);

  // Find the size of the Jacobian matrix.
  int jacobian_row_size = 0;
  // 在地图点中，找到需要剔除的帧上的特征点
  for (auto& item : map_server) {
    auto& feature = item.second;
    // Check how many camera states to be removed are associated
    // with this feature.
    vector<StateIDType> involved_cam_state_ids(0);
    for (const auto& cam_id : rm_cam_state_ids) {
      if (feature.observations.find(cam_id) !=
          feature.observations.end())
        involved_cam_state_ids.push_back(cam_id);
    }
    // 如果没有找到involved_cam_state_ids=0则不用边缘化
    if (involved_cam_state_ids.size() == 0) continue;
    // 如果只有一个，没法三角化，那就直接剔除
    if (involved_cam_state_ids.size() == 1) {
      feature.observations.erase(involved_cam_state_ids[0]);
      continue;
    }
    // 如果特征点没有三角化，则进行三角化
    if (!feature.is_initialized) {
      // Check if the feature can be initialize.
      // 是否可以进行三角化
      if (!feature.checkMotion(state_server.cam_states)) {
        // If the feature cannot be initialized, just remove
        // the observations associated with the camera states
        // to be removed.
        // 如果不能三角化，那直接剔除
        for (const auto& cam_id : involved_cam_state_ids)
          feature.observations.erase(cam_id);
        continue;
      } else {
        // 如果可以三角化，则进行三角化
        if(!feature.initializePosition(state_server.cam_states)) {
          // 如果不能三角化，就剔除
          for (const auto& cam_id : involved_cam_state_ids)
            feature.observations.erase(cam_id);
          continue;
        }
      }
    }
    // 计算雅克比矩阵行的维度
    jacobian_row_size += 4*involved_cam_state_ids.size() - 3;
  }

  //cout << "jacobian row #: " << jacobian_row_size << endl;

  // Compute the Jacobian and residual.
  MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
      21+6*state_server.cam_states.size());
  VectorXd r = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;
  // 遍历地图点，计算雅克比矩阵
  for (auto& item : map_server) {
    auto& feature = item.second;
    // Check how many camera states to be removed are associated
    // with this feature.
    vector<StateIDType> involved_cam_state_ids(0);
    for (const auto& cam_id : rm_cam_state_ids) {
      if (feature.observations.find(cam_id) !=
          feature.observations.end())
        involved_cam_state_ids.push_back(cam_id);
    }

    if (involved_cam_state_ids.size() == 0) continue;

    MatrixXd H_xj;
    VectorXd r_j;
    featureJacobian(feature.id, involved_cam_state_ids, H_xj, r_j);

    if (gatingTest(H_xj, r_j, involved_cam_state_ids.size())) {
      H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
      r.segment(stack_cntr, r_j.rows()) = r_j;
      stack_cntr += H_xj.rows();
    }
    // 然后从特征点的观测中剔除这需要边缘化的这两帧
    for (const auto& cam_id : involved_cam_state_ids)
      feature.observations.erase(cam_id);
  }

  H_x.conservativeResize(stack_cntr, H_x.cols());
  r.conservativeResize(stack_cntr);

  // Perform measurement update.
  // 滤波更新，更新状态量和协方差矩阵
  measurementUpdate(H_x, r);
  // 在从协方差矩阵中剔除这两帧信息
  for (const auto& cam_id : rm_cam_state_ids) {
    int cam_sequence = std::distance(state_server.cam_states.begin(),
        state_server.cam_states.find(cam_id));
    int cam_state_start = 21 + 6*cam_sequence;
    int cam_state_end = cam_state_start + 6;

    // Remove the corresponding rows and columns in the state
    // covariance matrix.
    if (cam_state_end < state_server.state_cov.rows()) {
      state_server.state_cov.block(cam_state_start, 0,
          state_server.state_cov.rows()-cam_state_end,
          state_server.state_cov.cols()) =
        state_server.state_cov.block(cam_state_end, 0,
            state_server.state_cov.rows()-cam_state_end,
            state_server.state_cov.cols());

      state_server.state_cov.block(0, cam_state_start,
          state_server.state_cov.rows(),
          state_server.state_cov.cols()-cam_state_end) =
        state_server.state_cov.block(0, cam_state_end,
            state_server.state_cov.rows(),
            state_server.state_cov.cols()-cam_state_end);

      state_server.state_cov.conservativeResize(
          state_server.state_cov.rows()-6, state_server.state_cov.cols()-6);
    } else {
      state_server.state_cov.conservativeResize(
          state_server.state_cov.rows()-6, state_server.state_cov.cols()-6);
    }

    // Remove this camera state in the state vector.
    state_server.cam_states.erase(cam_id);
  }

  return;
}

void MsckfVio::onlineReset() {

  // Never perform online reset if position std threshold
  // is non-positive.
  if (position_std_threshold <= 0) return;
  static long long int online_reset_counter = 0;

  // Check the uncertainty of positions to determine if
  // the system can be reset.
  double position_x_std = std::sqrt(state_server.state_cov(12, 12));
  double position_y_std = std::sqrt(state_server.state_cov(13, 13));
  double position_z_std = std::sqrt(state_server.state_cov(14, 14));

  if (position_x_std < position_std_threshold &&
      position_y_std < position_std_threshold &&
      position_z_std < position_std_threshold) return;

  ROS_WARN("Start %lld online reset procedure...",
      ++online_reset_counter);
  ROS_INFO("Stardard deviation in xyz: %f, %f, %f",
      position_x_std, position_y_std, position_z_std);

  // Remove all existing camera states.
  state_server.cam_states.clear();

  // Clear all exsiting features in the map.
  map_server.clear();

  // Reset the state covariance.
  double gyro_bias_cov, acc_bias_cov, velocity_cov;
  nh.param<double>("initial_covariance/velocity",
      velocity_cov, 0.25);
  nh.param<double>("initial_covariance/gyro_bias",
      gyro_bias_cov, 1e-4);
  nh.param<double>("initial_covariance/acc_bias",
      acc_bias_cov, 1e-2);

  double extrinsic_rotation_cov, extrinsic_translation_cov;
  nh.param<double>("initial_covariance/extrinsic_rotation_cov",
      extrinsic_rotation_cov, 3.0462e-4);
  nh.param<double>("initial_covariance/extrinsic_translation_cov",
      extrinsic_translation_cov, 1e-4);

  state_server.state_cov = MatrixXd::Zero(21, 21);
  for (int i = 3; i < 6; ++i)
    state_server.state_cov(i, i) = gyro_bias_cov;
  for (int i = 6; i < 9; ++i)
    state_server.state_cov(i, i) = velocity_cov;
  for (int i = 9; i < 12; ++i)
    state_server.state_cov(i, i) = acc_bias_cov;
  for (int i = 15; i < 18; ++i)
    state_server.state_cov(i, i) = extrinsic_rotation_cov;
  for (int i = 18; i < 21; ++i)
    state_server.state_cov(i, i) = extrinsic_translation_cov;

  ROS_WARN("%lld online reset complete...", online_reset_counter);
  return;
}

void MsckfVio::publish(const ros::Time& time) {

  // Convert the IMU frame to the body frame.
  const IMUState& imu_state = state_server.imu_state;
  Eigen::Isometry3d T_i_w = Eigen::Isometry3d::Identity();
  T_i_w.linear() = quaternionToRotation(
      imu_state.orientation).transpose();
  T_i_w.translation() = imu_state.position;

  Eigen::Isometry3d T_b_w = IMUState::T_imu_body * T_i_w *
    IMUState::T_imu_body.inverse();
  Eigen::Vector3d body_velocity =
    IMUState::T_imu_body.linear() * imu_state.velocity;

  // Publish tf
  if (publish_tf) {
    tf::Transform T_b_w_tf;
    tf::transformEigenToTF(T_b_w, T_b_w_tf);
    tf_pub.sendTransform(tf::StampedTransform(
          T_b_w_tf, time, fixed_frame_id, child_frame_id));
  }

  // Publish the odometry
  nav_msgs::Odometry odom_msg;
  odom_msg.header.stamp = time;
  odom_msg.header.frame_id = fixed_frame_id;
  odom_msg.child_frame_id = child_frame_id;

  tf::poseEigenToMsg(T_b_w, odom_msg.pose.pose);
  tf::vectorEigenToMsg(body_velocity, odom_msg.twist.twist.linear);

  // Convert the covariance.
  Matrix3d P_oo = state_server.state_cov.block<3, 3>(0, 0);
  Matrix3d P_op = state_server.state_cov.block<3, 3>(0, 12);
  Matrix3d P_po = state_server.state_cov.block<3, 3>(12, 0);
  Matrix3d P_pp = state_server.state_cov.block<3, 3>(12, 12);
  Matrix<double, 6, 6> P_imu_pose = Matrix<double, 6, 6>::Zero();
  P_imu_pose << P_pp, P_po, P_op, P_oo;

  Matrix<double, 6, 6> H_pose = Matrix<double, 6, 6>::Zero();
  H_pose.block<3, 3>(0, 0) = IMUState::T_imu_body.linear();
  H_pose.block<3, 3>(3, 3) = IMUState::T_imu_body.linear();
  Matrix<double, 6, 6> P_body_pose = H_pose *
    P_imu_pose * H_pose.transpose();

  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      odom_msg.pose.covariance[6*i+j] = P_body_pose(i, j);

  // Construct the covariance for the velocity.
  Matrix3d P_imu_vel = state_server.state_cov.block<3, 3>(6, 6);
  Matrix3d H_vel = IMUState::T_imu_body.linear();
  Matrix3d P_body_vel = H_vel * P_imu_vel * H_vel.transpose();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      odom_msg.twist.covariance[i*6+j] = P_body_vel(i, j);

  odom_pub.publish(odom_msg);

  // Publish the 3D positions of the features that
  // has been initialized.
  pcl::PointCloud<pcl::PointXYZ>::Ptr feature_msg_ptr(
      new pcl::PointCloud<pcl::PointXYZ>());
  feature_msg_ptr->header.frame_id = fixed_frame_id;
  feature_msg_ptr->height = 1;
  for (const auto& item : map_server) {
    const auto& feature = item.second;
    if (feature.is_initialized) {
      Vector3d feature_position =
        IMUState::T_imu_body.linear() * feature.position;
      feature_msg_ptr->points.push_back(pcl::PointXYZ(
            feature_position(0), feature_position(1), feature_position(2)));
    }
  }
  feature_msg_ptr->width = feature_msg_ptr->points.size();

  feature_pub.publish(feature_msg_ptr);

  return;
}

} // namespace msckf_vio

