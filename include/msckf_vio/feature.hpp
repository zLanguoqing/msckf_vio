/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_FEATURE_H
#define MSCKF_VIO_FEATURE_H

#include <iostream>
#include <map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include "math_utils.hpp"
#include "imu_state.h"
#include "cam_state.h"

namespace msckf_vio {

/*
 * @brief Feature Salient part of an image. Please refer
 *    to the Appendix of "A Multi-State Constraint Kalman
 *    Filter for Vision-aided Inertial Navigation" for how
 *    the 3d position of a feature is initialized.
 */
struct Feature {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef long long int FeatureIDType;

  /*
   * @brief OptimizationConfig Configuration parameters
   *    for 3d feature position optimization.
   */
  struct OptimizationConfig {
    double translation_threshold;
    double huber_epsilon;
    double estimation_precision;
    double initial_damping;
    int outer_loop_max_iteration;
    int inner_loop_max_iteration;

    OptimizationConfig():
      translation_threshold(0.2),
      huber_epsilon(0.01),
      estimation_precision(5e-7),
      initial_damping(1e-3),
      outer_loop_max_iteration(10),
      inner_loop_max_iteration(10) {
      return;
    }
  };

  // Constructors for the struct.
  Feature(): id(0), position(Eigen::Vector3d::Zero()),
    is_initialized(false) {}

  Feature(const FeatureIDType& new_id): id(new_id),
    position(Eigen::Vector3d::Zero()),
    is_initialized(false) {}

  /*
   * @brief cost Compute the cost of the camera observations
   * @param T_c0_c1 A rigid body transformation takes
   *    a vector in c0 frame to ci frame.
   * @param x The current estimation.
   * @param z The ith measurement of the feature j in ci frame.
   * @return e The cost of this observation.
   */
  inline void cost(const Eigen::Isometry3d& T_c0_ci,
      const Eigen::Vector3d& x, const Eigen::Vector2d& z,
      double& e) const;

  /*
   * @brief jacobian Compute the Jacobian of the camera observation
   * @param T_c0_c1 A rigid body transformation takes
   *    a vector in c0 frame to ci frame.
   * @param x The current estimation.
   * @param z The actual measurement of the feature in ci frame.
   * @return J The computed Jacobian.
   * @return r The computed residual.
   * @return w Weight induced by huber kernel.
   */
  inline void jacobian(const Eigen::Isometry3d& T_c0_ci,
      const Eigen::Vector3d& x, const Eigen::Vector2d& z,
      Eigen::Matrix<double, 2, 3>& J, Eigen::Vector2d& r,
      double& w) const;

  /*
   * @brief generateInitialGuess Compute the initial guess of
   *    the feature's 3d position using only two views.
   * @param T_c1_c2: A rigid body transformation taking
   *    a vector from c2 frame to c1 frame.
   * @param z1: feature observation in c1 frame.
   * @param z2: feature observation in c2 frame.
   * @return p: Computed feature position in c1 frame.
   */
  inline void generateInitialGuess(
      const Eigen::Isometry3d& T_c1_c2, const Eigen::Vector2d& z1,
      const Eigen::Vector2d& z2, Eigen::Vector3d& p) const;

  /*
   * @brief checkMotion Check the input camera poses to ensure
   *    there is enough translation to triangulate the feature
   *    positon.
   * @param cam_states : input camera poses.
   * @return True if the translation between the input camera
   *    poses is sufficient.
   */
  inline bool checkMotion(
      const CamStateServer& cam_states) const;

  /*
   * @brief InitializePosition Intialize the feature position
   *    based on all current available measurements.
   * @param cam_states: A map containing the camera poses with its
   *    ID as the associated key value.
   * @return The computed 3d position is used to set the position
   *    member variable. Note the resulted position is in world
   *    frame.
   * @return True if the estimated 3d position of the feature
   *    is valid.
   */
  inline bool initializePosition(
      const CamStateServer& cam_states);


  // An unique identifier for the feature.
  // In case of long time running, the variable
  // type of id is set to FeatureIDType in order
  // to avoid duplication.
  FeatureIDType id;

  // id for next feature
  static FeatureIDType next_id;

  // Store the observations of the features in the
  // state_id(key)-image_coordinates(value) manner.
  std::map<StateIDType, Eigen::Vector4d, std::less<StateIDType>,
    Eigen::aligned_allocator<
      std::pair<const StateIDType, Eigen::Vector4d> > > observations;

  // 3d postion of the feature in the world frame.
  Eigen::Vector3d position;

  // A indicator to show if the 3d postion of the feature
  // has been initialized or not.
  bool is_initialized;

  // Noise for a normalized feature measurement.
  static double observation_noise;

  // Optimization configuration for solving the 3d position.
  static OptimizationConfig optimization_config;

};

typedef Feature::FeatureIDType FeatureIDType;
typedef std::map<FeatureIDType, Feature, std::less<int>,
        Eigen::aligned_allocator<
        std::pair<const FeatureIDType, Feature> > > MapServer;

/**
 * @brief 边缘化丢失的3D点进行ekf更新，得到系统状态估计值
 * @param T_c0_ci 相机相对位姿
 * @param x       归一化坐标
 * @param z       观测值
 * @param e       残差
 * @return  void
 */
void Feature::cost(const Eigen::Isometry3d& T_c0_ci,
    const Eigen::Vector3d& x, const Eigen::Vector2d& z,
    double& e) const {
  // Compute hi1, hi2, and hi3 as Equation (37).
  const double& alpha = x(0);
  const double& beta = x(1);
  const double& rho = x(2);
  // 求得第一帧的归一化坐标在最后一帧的的观测坐标
  Eigen::Vector3d h = T_c0_ci.linear()*
    Eigen::Vector3d(alpha, beta, 1.0) + rho*T_c0_ci.translation();
  double& h1 = h(0);
  double& h2 = h(1);
  double& h3 = h(2);

  // Predict the feature observation in ci frame.
  // 并归一化
  Eigen::Vector2d z_hat(h1/h3, h2/h3);

  // Compute the residual.
  // 求残差值
  e = (z_hat-z).squaredNorm();
  return;
}

void Feature::jacobian(const Eigen::Isometry3d& T_c0_ci,
    const Eigen::Vector3d& x, const Eigen::Vector2d& z,
    Eigen::Matrix<double, 2, 3>& J, Eigen::Vector2d& r,
    double& w) const {

  // Compute hi1, hi2, and hi3 as Equation (37).
  const double& alpha = x(0);
  const double& beta = x(1);
  const double& rho = x(2);

  Eigen::Vector3d h = T_c0_ci.linear()*
    Eigen::Vector3d(alpha, beta, 1.0) + rho*T_c0_ci.translation();
  double& h1 = h(0);
  double& h2 = h(1);
  double& h3 = h(2);

  // Compute the Jacobian.
  Eigen::Matrix3d W;
  W.leftCols<2>() = T_c0_ci.linear().leftCols<2>();
  W.rightCols<1>() = T_c0_ci.translation();

  J.row(0) = 1/h3*W.row(0) - h1/(h3*h3)*W.row(2);
  J.row(1) = 1/h3*W.row(1) - h2/(h3*h3)*W.row(2);

  // Compute the residual.
  Eigen::Vector2d z_hat(h1/h3, h2/h3);
  r = z_hat - z;

  // Compute the weight based on the residual.
  double e = r.norm();
  if (e <= optimization_config.huber_epsilon)
    w = 1.0;
  else
    w = std::sqrt(2.0*optimization_config.huber_epsilon / e);

  return;
}
/**
 * @brief   三角化初始值计算，用于高斯牛顿优化
 * @param   T_c1_c2        为最后一帧到第一帧的相对位姿
 * @param   z1             第一次观测
 * @param   z2             最后一次观测
 * @param   p              初始化坐标值
 * @return  void
 */
void Feature::generateInitialGuess(
    const Eigen::Isometry3d& T_c1_c2, const Eigen::Vector2d& z1,
    const Eigen::Vector2d& z2, Eigen::Vector3d& p) const {
  // Construct a least square problem to solve the depth.
  Eigen::Vector3d m = T_c1_c2.linear() * Eigen::Vector3d(z1(0), z1(1), 1.0);

  Eigen::Vector2d A(0.0, 0.0);
  A(0) = m(0) - z2(0)*m(2);
  A(1) = m(1) - z2(1)*m(2);

  Eigen::Vector2d b(0.0, 0.0);
  b(0) = z2(0)*T_c1_c2.translation()(2) - T_c1_c2.translation()(0);
  b(1) = z2(1)*T_c1_c2.translation()(2) - T_c1_c2.translation()(1);

  // Solve for the depth.
  double depth = (A.transpose() * A).inverse() * A.transpose() * b;
  p(0) = z1(0) * depth;
  p(1) = z1(1) * depth;
  p(2) = depth;
  return;
}
/**
 * @brief 判断是否有足够的移动，能够三角化
 * @param cam_states 相机状态
 * @return  bool
 */
bool Feature::checkMotion(
    const CamStateServer& cam_states) const {
  // 观测信息的第一帧和最后一帧
  const StateIDType& first_cam_id = observations.begin()->first;
  const StateIDType& last_cam_id = (--observations.end())->first;
  // 观测信息的第一帧相机的对应的位姿，linear代表旋转量，表示cam 2 world
  Eigen::Isometry3d first_cam_pose;
  first_cam_pose.linear() = quaternionToRotation(
      cam_states.find(first_cam_id)->second.orientation).transpose();
  // translation 表示相机的位移 cam 2 world
  first_cam_pose.translation() =
    cam_states.find(first_cam_id)->second.position;
// 这里是最后一帧的图像信息
  Eigen::Isometry3d last_cam_pose;
  last_cam_pose.linear() = quaternionToRotation(
      cam_states.find(last_cam_id)->second.orientation).transpose();
  last_cam_pose.translation() =
    cam_states.find(last_cam_id)->second.position;

  // Get the direction of the feature when it is first observed.
  // This direction is represented in the world frame.
  // 计算相机到世界坐标系下的点的向量
  Eigen::Vector3d feature_direction(
      observations.begin()->second(0),
      observations.begin()->second(1), 1.0);
  feature_direction = feature_direction / feature_direction.norm();
  feature_direction = first_cam_pose.linear()*feature_direction;

  // Compute the translation between the first frame
  // and the last frame. We assume the first frame and
  // the last frame will provide the largest motion to
  // speed up the checking process.
  // 计算两个相机的相对位移
  Eigen::Vector3d translation = last_cam_pose.translation() -
    first_cam_pose.translation();
  // 计算两个相机位移投影到feature_direction方向上
  double parallel_translation =
    translation.transpose()*feature_direction;
  // 根据向量运算法则，orthogonal_translation为与feature_direction垂直方向的位移  
  Eigen::Vector3d orthogonal_translation = translation -
    parallel_translation*feature_direction;
  // orthogonal_translation的模长小于阈值，则认为是满足三角化条件
  if (orthogonal_translation.norm() >
      optimization_config.translation_threshold)
    return true;
  else return false;
}
/**
 * @brief 三角化，采用基于几何的非线性误差的估计，包括初始值计算和非线性优化
 * @param cam_states 相机状态
 * @return  bool
 */
bool Feature::initializePosition(
    const CamStateServer& cam_states) {
  // Organize camera poses and feature observations properly.
  std::vector<Eigen::Isometry3d,
    Eigen::aligned_allocator<Eigen::Isometry3d> > cam_poses(0);
  std::vector<Eigen::Vector2d,
    Eigen::aligned_allocator<Eigen::Vector2d> > measurements(0);
  // 遍历一个特征点的观测
  for (auto& m : observations) {
    // TODO: This should be handled properly. Normally, the
    //    required camera states should all be available in
    //    the input cam_states buffer.
    // cam_state_iter为相机状态中的迭代个数
    // m.first 为特征点观测中对应的相机状态ID
    auto cam_state_iter = cam_states.find(m.first);
    // 如果没找到，就继续
    if (cam_state_iter == cam_states.end()) continue;

    // Add the measurement.
    // 将观测信息放入measurements中
    measurements.push_back(m.second.head<2>());
    measurements.push_back(m.second.tail<2>());

    // This camera pose will take a vector from this camera frame
    // to the world frame.

    Eigen::Isometry3d cam0_pose;
    cam0_pose.linear() = quaternionToRotation(
        cam_state_iter->second.orientation).transpose();
    cam0_pose.translation() = cam_state_iter->second.position;

    Eigen::Isometry3d cam1_pose;
    cam1_pose = cam0_pose * CAMState::T_cam0_cam1.inverse();
    // 将左右相机的位姿放入cam_poses
    cam_poses.push_back(cam0_pose);
    cam_poses.push_back(cam1_pose);
  }

  // All camera poses should be modified such that it takes a
  // vector from the first camera frame in the buffer to this
  // camera frame.
  // 计算第一次观测到最后一次观测的相对位姿关系

  Eigen::Isometry3d T_c0_w = cam_poses[0];
  for (auto& pose : cam_poses)
    pose = pose.inverse() * T_c0_w;

  // Generate initial guess
  // 设定初始化位姿
  Eigen::Vector3d initial_position(0.0, 0.0, 0.0);
  // cam_poses[cam_poses.size()-1] 为最后一帧到第一帧的相对位姿
  // measurements[0] 第一次观测
  // measurements[measurements.size()-1]，最后一次观测
  // initial_position 初始化坐标值
  generateInitialGuess(cam_poses[cam_poses.size()-1], measurements[0],
      measurements[measurements.size()-1], initial_position);
  // initial_position 最后输出为初始值
  Eigen::Vector3d solution(
      initial_position(0)/initial_position(2),
      initial_position(1)/initial_position(2),
      1.0/initial_position(2));

  // Apply Levenberg-Marquart method to solve for the 3d position.
  // LM算法
  double lambda = optimization_config.initial_damping;
  int inner_loop_cntr = 0;
  int outer_loop_cntr = 0;
  bool is_cost_reduced = false;
  double delta_norm = 0;

  // Compute the initial cost.
  double total_cost = 0.0;
  for (int i = 0; i < cam_poses.size(); ++i) {
    double this_cost = 0.0;
    // 计算每个观测的代价cost，并累加起来
    cost(cam_poses[i], solution, measurements[i], this_cost);
    total_cost += this_cost;
  }

  // Outer loop.
  do {
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();

    for (int i = 0; i < cam_poses.size(); ++i) {
      Eigen::Matrix<double, 2, 3> J;
      Eigen::Vector2d r;
      double w;
      // 计算雅克比矩阵，
      jacobian(cam_poses[i], solution, measurements[i], J, r, w);
      // 因为LM算法需要(JTJ+lamta×I)deltax=b
      if (w == 1) {
        A += J.transpose() * J;
        b += J.transpose() * r;
      } else {
        double w_square = w * w;
        A += w_square * J.transpose() * J;
        b += w_square * J.transpose() * r;
      }
    }

    // Inner loop.
    // Solve for the delta that can reduce the total cost.
    do {
      Eigen::Matrix3d damper = lambda * Eigen::Matrix3d::Identity();
      // (JTJ+I)deltax=b
      // 求解deltax
      Eigen::Vector3d delta = (A+damper).ldlt().solve(b);
      Eigen::Vector3d new_solution = solution - delta;
      delta_norm = delta.norm();
      // 每求解一次小重新计算一次代价是否变小
      double new_cost = 0.0;
      for (int i = 0; i < cam_poses.size(); ++i) {
        double this_cost = 0.0;
        cost(cam_poses[i], new_solution, measurements[i], this_cost);
        new_cost += this_cost;
      }
      // 如果变小，说明在靠近真值，则需要用高斯牛顿的方法进行优化，因为高斯牛顿优化在接近真值的时候是线性收敛的
      if (new_cost < total_cost) {
        is_cost_reduced = true;
        solution = new_solution;
        total_cost = new_cost;
        lambda = lambda/10 > 1e-10 ? lambda/10 : 1e-10;
      } else {
        // 如果增大，则认为不靠近真值，需要用梯度下降法快速靠近真值把lambda变大
        is_cost_reduced = false;
        lambda = lambda*10 < 1e12 ? lambda*10 : 1e12;
      }
      // inner_loop_cntr 为迭代次数，满足最大迭代值则自动收敛
      // is_cost_reduced为false停止迭代
    } while (inner_loop_cntr++ <
        optimization_config.inner_loop_max_iteration && !is_cost_reduced);

    inner_loop_cntr = 0;

  } while (outer_loop_cntr++ <
      optimization_config.outer_loop_max_iteration &&
      delta_norm > optimization_config.estimation_precision);

  // Covert the feature position from inverse depth
  // representation to its 3d coordinate.
  Eigen::Vector3d final_position(solution(0)/solution(2),
      solution(1)/solution(2), 1.0/solution(2));

  // Check if the solution is valid. Make sure the feature
  // is in front of every camera frame observing it.
  bool is_valid_solution = true;
  //判断计算出来的坐标值是否有问题
  for (const auto& pose : cam_poses) {
    Eigen::Vector3d position =
      pose.linear()*final_position + pose.translation();
    if (position(2) <= 0) {
      is_valid_solution = false;
      break;
    }
  }

  // Convert the feature position to the world frame.
  // 计算世界坐标系的坐标值
  position = T_c0_w.linear()*final_position + T_c0_w.translation();

  if (is_valid_solution)
    is_initialized = true;

  return is_valid_solution;
}
} // namespace msckf_vio

#endif // MSCKF_VIO_FEATURE_H
