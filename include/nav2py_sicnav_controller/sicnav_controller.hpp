/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SICNav Controller for nav2py
 */
#ifndef NAV2PY_SICNAV_CONTROLLER__SICNAV_CONTROLLER_HPP_
#define NAV2PY_SICNAV_CONTROLLER__SICNAV_CONTROLLER_HPP_

#include <string>
#include <mutex>
#include "nav2py/controller.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "rclcpp_lifecycle/lifecycle_publisher.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "tf2_ros/buffer.h"

namespace nav2py_sicnav_controller
{
  class SicnavController : public nav2py::Controller
  {
  public:
    SicnavController();

    void configure(
        const rclcpp_lifecycle::LifecycleNode::WeakPtr &parent,
        std::string name,
        const std::shared_ptr<tf2_ros::Buffer> tf,
        const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

    void cleanup() override;
    void activate() override;
    void deactivate() override;

    geometry_msgs::msg::TwistStamped computeVelocityCommands(
        const geometry_msgs::msg::PoseStamped &pose,
        const geometry_msgs::msg::Twist &velocity,
        nav2_core::GoalChecker *goal_checker) override;

    void setPlan(const nav_msgs::msg::Path &path) override;
    void setSpeedLimit(const double &speed_limit, const bool &percentage) override;

  protected:
    void sendData(
        const geometry_msgs::msg::PoseStamped &pose,
        const geometry_msgs::msg::Twist &velocity);

    void sendScan(const sensor_msgs::msg::LaserScan::SharedPtr scan);

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);

    bool transformPose(
        const std::shared_ptr<tf2_ros::Buffer> tf,
        const std::string frame,
        const geometry_msgs::msg::PoseStamped &in_pose,
        geometry_msgs::msg::PoseStamped &out_pose,
        const double transform_tolerance_secs) const;

    bool isValidCmdVel(const geometry_msgs::msg::TwistStamped &cmd_vel);
    geometry_msgs::msg::TwistStamped smoothVelocity(const geometry_msgs::msg::TwistStamped &cmd_vel);

    rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
    std::shared_ptr<tf2_ros::Buffer> tf_;
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
    std::string plugin_name_;
    rclcpp::Logger logger_{rclcpp::get_logger("SicnavController")};
    rclcpp::Clock::SharedPtr clock_;
    rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Path>::SharedPtr global_pub_;
    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_pub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;
    nav_msgs::msg::Path global_plan_;
    geometry_msgs::msg::TwistStamped last_cmd_vel_;
    geometry_msgs::msg::TwistStamped prev_cmd_vel_;
    rclcpp::Duration transform_tolerance_{0, 0};
    std::mutex cmd_vel_mutex_;
    double max_speed_;
    double neighbor_dist_;
    double time_horizon_;
    double smoothing_factor_;
    double max_angular_speed_;
    double safety_threshold_;
  };
} // namespace nav2py_sicnav_controller

#endif // NAV2PY_SICNAV_CONTROLLER__SICNAV_CONTROLLER_HPP_