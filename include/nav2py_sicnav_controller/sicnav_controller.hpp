/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SICNav Controller for nav2py
 */
#ifndef NAV2PY_SICNAV_CONTROLLER__SICNAV_CONTROLLER_HPP_
#define NAV2PY_SICNAV_CONTROLLER__SICNAV_CONTROLLER_HPP_

#include <string>
#include <vector>
#include <memory>
#include "nav2py/controller.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "tf2_ros/buffer.h"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"

namespace sicnav_controller
{
  class SicnavController : public nav2py::Controller
  {
  public:
    SicnavController() = default;
    ~SicnavController() override = default;

    void configure(
        const rclcpp_lifecycle::LifecycleNode::WeakPtr &parent,
        std::string name, const std::shared_ptr<tf2_ros::Buffer> tf,
        const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

    void cleanup() override;
    void activate() override;
    void deactivate() override;
    void setSpeedLimit(const double &speed_limit, const bool &percentage) override;

    geometry_msgs::msg::TwistStamped computeVelocityCommands(
        const geometry_msgs::msg::PoseStamped &pose,
        const geometry_msgs::msg::Twist &velocity,
        nav2_core::GoalChecker *goal_checker) override;

    void setPlan(const nav_msgs::msg::Path &path) override;

  protected:
    void sendData(
        const geometry_msgs::msg::PoseStamped &pose,
        const geometry_msgs::msg::Twist &velocity);
    void sendScan(const sensor_msgs::msg::LaserScan::SharedPtr scan);
    bool transformPose(
        const std::shared_ptr<tf2_ros::Buffer> tf,
        const std::string frame,
        const geometry_msgs::msg::PoseStamped &in_pose,
        geometry_msgs::msg::PoseStamped &out_pose,
        const rclcpp::Duration &transform_tolerance) const;
    bool isValidCmdVel(const geometry_msgs::msg::TwistStamped &cmd_vel);
    geometry_msgs::msg::TwistStamped smoothVelocity(const geometry_msgs::msg::TwistStamped &cmd_vel);

    rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
    std::shared_ptr<tf2_ros::Buffer> tf_;
    std::string plugin_name_;
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
    rclcpp::Logger logger_{rclcpp::get_logger("SicnavController")};
    rclcpp::Clock::SharedPtr clock_;

    rclcpp::Duration transform_tolerance_{0, 0};
    nav_msgs::msg::Path global_plan_;
    std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Path>> global_pub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::LaserScan>> scan_pub_;

    // Parameters
    double max_speed_;
    double neighbor_dist_;
    double time_horizon_;
    double smoothing_factor_;
    double max_angular_speed_;
    double safety_threshold_;
    rclcpp::ParameterCallbackHandle::SharedPtr parameter_callback_handle_;

    // State
    geometry_msgs::msg::TwistStamped last_cmd_vel_;
    geometry_msgs::msg::TwistStamped prev_cmd_vel_;
    std::mutex cmd_vel_mutex_;
  };
} // namespace sicnav_controller

#endif // NAV2PY_SICNAV_CONTROLLER__SICNAV_CONTROLLER_HPP_