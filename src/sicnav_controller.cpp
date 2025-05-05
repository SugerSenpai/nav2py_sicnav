/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * SICNav Controller for nav2py
 */
#include <algorithm>
#include <string>
#include <memory>
#include <stdexcept>
#include <chrono>
#include "nav2_core/exceptions.hpp"
#include "nav2_util/node_utils.hpp"
#include "nav2py_sicnav_controller/sicnav_controller.hpp"
#include "nav2_costmap_2d/footprint_collision_checker.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "nav_msgs/msg/detail/path__traits.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav2py/controller.hpp"
#include "nav2py/utils.hpp"
#include "pluginlib/class_list_macros.hpp"

using nav2_util::declare_parameter_if_not_declared;

namespace nav2py_sicnav_controller
{
  SicnavController::SicnavController() = default;

  void SicnavController::configure(
      const rclcpp_lifecycle::LifecycleNode::WeakPtr &parent,
      std::string name, const std::shared_ptr<tf2_ros::Buffer> tf,
      const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
  {
    node_ = parent;
    auto node = node_.lock();
    costmap_ros_ = costmap_ros;
    tf_ = tf;
    plugin_name_ = name;
    logger_ = node->get_logger();
    clock_ = node->get_clock();

    // Declare parameters
    declare_parameter_if_not_declared(node, plugin_name_ + ".transform_tolerance", rclcpp::ParameterValue(0.1));
    declare_parameter_if_not_declared(node, plugin_name_ + ".max_speed", rclcpp::ParameterValue(0.5));
    declare_parameter_if_not_declared(node, plugin_name_ + ".neighbor_dist", rclcpp::ParameterValue(5.0));
    declare_parameter_if_not_declared(node, plugin_name_ + ".time_horizon", rclcpp::ParameterValue(5.0));
    declare_parameter_if_not_declared(node, plugin_name_ + ".smoothing_factor", rclcpp::ParameterValue(0.3));
    declare_parameter_if_not_declared(node, plugin_name_ + ".max_angular_speed", rclcpp::ParameterValue(1.0));
    declare_parameter_if_not_declared(node, plugin_name_ + ".safety_threshold", rclcpp::ParameterValue(180.0));

    // Get parameters
    double transform_tolerance_secs;
    node->get_parameter(plugin_name_ + ".transform_tolerance", transform_tolerance_secs);
    transform_tolerance_ = rclcpp::Duration::from_seconds(transform_tolerance_secs);
    node->get_parameter(plugin_name_ + ".max_speed", max_speed_);
    node->get_parameter(plugin_name_ + ".neighbor_dist", neighbor_dist_);
    node->get_parameter(plugin_name_ + ".time_horizon", time_horizon_);
    node->get_parameter(plugin_name_ + ".smoothing_factor", smoothing_factor_);
    node->get_parameter(plugin_name_ + ".max_angular_speed", max_angular_speed_);
    node->get_parameter(plugin_name_ + ".safety_threshold", safety_threshold_);

    // Validate parameters
    if (max_speed_ <= 0.0)
    {
      RCLCPP_WARN(logger_, "max_speed must be positive, setting to 0.5");
      max_speed_ = 0.5;
    }
    if (neighbor_dist_ <= 0.0)
    {
      RCLCPP_WARN(logger_, "neighbor_dist must be positive, setting to 5.0");
      neighbor_dist_ = 5.0;
    }
    if (time_horizon_ <= 0.0)
    {
      RCLCPP_WARN(logger_, "time_horizon must be positive, setting to 5.0");
      time_horizon_ = 5.0;
    }
    if (smoothing_factor_ < 0.0 || smoothing_factor_ > 1.0)
    {
      RCLCPP_WARN(logger_, "smoothing_factor must be in [0,1], setting to 0.3");
      smoothing_factor_ = 0.3;
    }
    if (max_angular_speed_ <= 0.0)
    {
      RCLCPP_WARN(logger_, "max_angular_speed must be positive, setting to 1.0");
      max_angular_speed_ = 1.0;
    }

    // Parameter callback
    auto parameter_callback = [this](const std::vector<rclcpp::Parameter> &parameters) -> rcl_interfaces::msg::SetParametersResult
    {
      rcl_interfaces::msg::SetParametersResult result;
      result.successful = true;
      for (const auto &param : parameters)
      {
        if (param.get_name() == plugin_name_ + ".max_speed" && param.as_double() > 0.0)
        {
          max_speed_ = param.as_double();
        }
        else if (param.get_name() == plugin_name_ + ".neighbor_dist" && param.as_double() > 0.0)
        {
          neighbor_dist_ = param.as_double();
        }
        else if (param.get_name() == plugin_name_ + ".time_horizon" && param.as_double() > 0.0)
        {
          time_horizon_ = param.as_double();
        }
        else if (param.get_name() == plugin_name_ + ".smoothing_factor" && param.as_double() >= 0.0 && param.as_double() <= 1.0)
        {
          smoothing_factor_ = param.as_double();
        }
        else if (param.get_name() == plugin_name_ + ".max_angular_speed" && param.as_double() > 0.0)
        {
          max_angular_speed_ = param.as_double();
        }
        else if (param.get_name() == plugin_name_ + ".transform_tolerance" && param.as_double() > 0.0)
        {
          transform_tolerance_ = rclcpp::Duration::from_seconds(param.as_double());
        }
        else
        {
          result.successful = false;
          result.reason = "Invalid parameter value";
        }
      }
      return result;
    };
    parameter_callback_handle_ = node->add_on_set_parameters_callback(parameter_callback);

    // Initialize nav2py
    try
    {
      std::string nav2py_script = ament_index_cpp::get_package_share_directory("nav2py_sicnav_controller") + "/../../lib/nav2py_sicnav_controller/nav2py_run";
      RCLCPP_INFO(logger_, "Attempting to initialize nav2py with script: %s", nav2py_script.c_str());
      nav2py_bootstrap(nav2py_script + " --host 127.0.0.1 --port 0");
      RCLCPP_INFO(logger_, "Initialized nav2py successfully");
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(logger_, "Failed to initialize nav2py: %s", e.what());
      throw std::runtime_error("Failed to initialize nav2py");
    }

    // Create publishers
    global_pub_ = node->create_publisher<nav_msgs::msg::Path>("received_global_plan", 1);
    scan_pub_ = node->create_publisher<sensor_msgs::msg::LaserScan>("sicnav/scan", 10);

    // Set up LaserScan subscription
    try
    {
      nav2py::utils::Costmap costmap(
          std::string(node->get_namespace()) + "/local_costmap/local_costmap",
          node->get_namespace());
      auto laserscan_observation = costmap.findObservationByType("LaserScan");
      if (laserscan_observation.has_value())
      {
        std::string topic = laserscan_observation.value().topic();
        RCLCPP_INFO(logger_, "Laser scan topic found: %s", topic.c_str());
        scan_sub_ = node->create_subscription<sensor_msgs::msg::LaserScan>(
            topic, rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile(),
            std::bind(&SicnavController::sendScan, this, std::placeholders::_1));
      }
      else
      {
        RCLCPP_WARN(logger_, "No laser scan topic found, falling back to /scan");
        scan_sub_ = node->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile(),
            std::bind(&SicnavController::sendScan, this, std::placeholders::_1));
      }
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(logger_, "Error setting up LaserScan subscription: %s", e.what());
      scan_sub_ = node->create_subscription<sensor_msgs::msg::LaserScan>(
          "/scan", rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile(),
          std::bind(&SicnavController::sendScan, this, std::placeholders::_1));
    }

    // Set up Odometry subscription
    try
    {
      odom_sub_ = node->create_subscription<nav_msgs::msg::Odometry>(
          "/task_generator_node/jackal/odom", rclcpp::SensorDataQoS(),
          std::bind(&SicnavController::odomCallback, this, std::placeholders::_1));
      RCLCPP_INFO(logger_, "Subscribed to odometry topic: /task_generator_node/jackal/odom");
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(logger_, "Error setting up odometry subscription: %s", e.what());
    }

    // Initialize state
    last_cmd_vel_.header.frame_id = "";
    prev_cmd_vel_.header.frame_id = "";

    RCLCPP_INFO(
        logger_,
        "Configured controller: %s of type nav2py_sicnav_controller::SicnavController with "
        "max_speed: %f, neighbor_dist: %f, time_horizon: %f, smoothing_factor: %f, max_angular_speed: %f",
        plugin_name_.c_str(), max_speed_, neighbor_dist_, time_horizon_, smoothing_factor_, max_angular_speed_);
  }

  void SicnavController::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    try
    {
      std::string odom_yaml = nav_msgs::msg::to_yaml(*msg, true);
      nav2py_send("odom", {odom_yaml});
      RCLCPP_DEBUG(logger_, "Sent odometry data to Python controller");
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(logger_, "Error sending odometry: %s", e.what());
    }
  }

  void SicnavController::sendData(
      const geometry_msgs::msg::PoseStamped &pose,
      const geometry_msgs::msg::Twist &velocity)
  {
    static int frame_count = 0;
    frame_count++;

    // Create structured data message
    std::stringstream ss;
    ss << "frame_info:\n";
    ss << "  id: " << frame_count << "\n";
    ss << "  timestamp: " << clock_->now().nanoseconds() << "\n";
    ss << "robot_pose:\n";
    ss << "  position:\n";
    ss << "    x: " << pose.pose.position.x << "\n";
    ss << "    y: " << pose.pose.position.y << "\n";
    ss << "    z: " << pose.pose.position.z << "\n";
    ss << "  orientation:\n";
    ss << "    x: " << pose.pose.orientation.x << "\n";
    ss << "    y: " << pose.pose.orientation.y << "\n";
    ss << "    z: " << pose.pose.orientation.z << "\n";
    ss << "    w: " << pose.pose.orientation.w << "\n";
    ss << "robot_velocity:\n";
    ss << "  linear:\n";
    ss << "    x: " << velocity.linear.x << "\n";
    ss << "    y: " << velocity.linear.y << "\n";
    ss << "    z: " << velocity.linear.z << "\n";
    ss << "  angular:\n";
    ss << "    x: " << velocity.angular.x << "\n";
    ss << "    y: " << velocity.angular.y << "\n";
    ss << "    z: " << velocity.angular.z << "\n";

    try
    {
      nav2py_send("data", {ss.str()});
      RCLCPP_DEBUG(logger_, "Data sent to Python side");
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(logger_, "Error sending data: %s", e.what());
    }
  }

  void SicnavController::sendScan(const sensor_msgs::msg::LaserScan::SharedPtr scan)
  {
    try
    {
      scan_pub_->publish(*scan);
      std::string scan_yaml = sensor_msgs::msg::to_yaml(*scan, true);
      nav2py_send("scan", {scan_yaml});
      RCLCPP_DEBUG(logger_, "Sent LaserScan data to Python controller");
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(logger_, "Error sending LaserScan: %s", e.what());
    }
  }

  void SicnavController::cleanup()
  {
    RCLCPP_INFO(logger_, "Cleaning up controller: %s", plugin_name_.c_str());
    try
    {
      nav2py_cleanup();
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(logger_, "Error during nav2py cleanup: %s", e.what());
    }
    global_pub_.reset();
    scan_pub_.reset();
    scan_sub_.reset();
    odom_sub_.reset();
    parameter_callback_handle_.reset();
  }

  void SicnavController::activate()
  {
    RCLCPP_INFO(logger_, "Activating controller: %s", plugin_name_.c_str());
    global_pub_->on_activate();
    scan_pub_->on_activate();
  }

  void SicnavController::deactivate()
  {
    RCLCPP_INFO(logger_, "Deactivating controller: %s", plugin_name_.c_str());
    global_pub_->on_deactivate();
    scan_pub_->on_deactivate();
  }

  void SicnavController::setSpeedLimit(const double &speed_limit, const bool &percentage)
  {
    std::lock_guard<std::mutex> lock(cmd_vel_mutex_);
    try
    {
      nav2py_send("speed_limit", {std::to_string(speed_limit), percentage ? "true" : "false"});
      RCLCPP_INFO(logger_, "Speed limit set to %f (percentage: %s)", speed_limit, percentage ? "true" : "false");
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(logger_, "Error sending speed limit: %s", e.what());
    }
  }

  geometry_msgs::msg::TwistStamped SicnavController::computeVelocityCommands(
      const geometry_msgs::msg::PoseStamped &pose,
      const geometry_msgs::msg::Twist &velocity,
      nav2_core::GoalChecker *goal_checker)
  {
    (void)goal_checker;

    // Transform pose to global frame
    geometry_msgs::msg::PoseStamped transformed_pose;
    std::string global_frame = costmap_ros_->getGlobalFrameID();
    if (!transformPose(tf_, global_frame, pose, transformed_pose, transform_tolerance_.seconds()))
    {
      RCLCPP_ERROR(logger_, "Failed to transform pose to %s frame", global_frame.c_str());
      return last_cmd_vel_;
    }

    // Transform velocity (linear components only)
    geometry_msgs::msg::Twist transformed_velocity = velocity;
    if (pose.header.frame_id != global_frame)
    {
      try
      {
        geometry_msgs::msg::Vector3Stamped vel_in, vel_out;
        vel_in.header = pose.header;
        vel_in.vector.x = velocity.linear.x;
        vel_in.vector.y = velocity.linear.y;
        tf2::doTransform(vel_in, vel_out, tf_->lookupTransform(global_frame, pose.header.frame_id, tf2::TimePointZero));
        transformed_velocity.linear.x = vel_out.vector.x;
        transformed_velocity.linear.y = vel_out.vector.y;
      }
      catch (tf2::TransformException &ex)
      {
        RCLCPP_ERROR(logger_, "Failed to transform velocity: %s", ex.what());
        return last_cmd_vel_;
      }
    }

    // Send data to Python
    try
    {
      sendData(transformed_pose, transformed_velocity);
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(logger_, "Error sending data: %s", e.what());
    }

    geometry_msgs::msg::TwistStamped cmd_vel;
    cmd_vel.header.frame_id = pose.header.frame_id;
    cmd_vel.header.stamp = clock_->now();

    try
    {
      RCLCPP_DEBUG(logger_, "Waiting for velocity command from Python...");
      // Add timeout (1 second) to prevent hanging
      auto start_time = clock_->now();
      while (!this->has_cmd_vel() && (clock_->now() - start_time).seconds() < 1.0)
      {
        rclcpp::sleep_for(std::chrono::milliseconds(10));
      }
      if (this->has_cmd_vel())
      {
        cmd_vel.twist = this->wait_for_cmd_vel();
        RCLCPP_INFO(
            logger_,
            "Received velocity command: linear_x=%.2f, angular_z=%.2f",
            cmd_vel.twist.linear.x, cmd_vel.twist.angular.z);
      }
      else
      {
        throw std::runtime_error("Timeout waiting for velocity command");
      }
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(logger_, "Error receiving velocity command: %s", e.what());
      cmd_vel.twist.linear.x = 0.0;
      cmd_vel.twist.linear.y = 0.0;
      cmd_vel.twist.angular.z = 0.0;
    }

    // Validate and smooth velocity
    std::lock_guard<std::mutex> lock(cmd_vel_mutex_);
    if (isValidCmdVel(cmd_vel))
    {
      last_cmd_vel_ = smoothVelocity(cmd_vel);
    }
    else
    {
      last_cmd_vel_.twist.linear.x = 0.0;
      last_cmd_vel_.twist.linear.y = 0.0;
      last_cmd_vel_.twist.angular.z = 0.0;
    }
    return last_cmd_vel_;
  }

  void SicnavController::setPlan(const nav_msgs::msg::Path &path)
  {
    global_plan_ = path;
    global_pub_->publish(path);
    try
    {
      std::string path_yaml = nav_msgs::msg::to_yaml(path, true);
      nav2py_send("path", {path_yaml});
      RCLCPP_INFO(logger_, "Sent path data to Python controller");
    }
    catch (const std::exception &e)
    {
      RCLCPP_ERROR(logger_, "Error sending path: %s", e.what());
    }
  }

  bool SicnavController::transformPose(
      const std::shared_ptr<tf2_ros::Buffer> tf,
      const std::string frame,
      const geometry_msgs::msg::PoseStamped &in_pose,
      geometry_msgs::msg::PoseStamped &out_pose,
      const double transform_tolerance_secs) const
  {
    try
    {
      tf2::Duration tf_timeout(std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(transform_tolerance_secs)));
      out_pose = tf->transform(in_pose, frame, tf_timeout);
      return true;
    }
    catch (tf2::TransformException &ex)
    {
      RCLCPP_ERROR(logger_, "Transform failed: %s", ex.what());
      return false;
    }
  }

  bool SicnavController::isValidCmdVel(const geometry_msgs::msg::TwistStamped &cmd_vel)
  {
    // Check velocity limits
    if (std::abs(cmd_vel.twist.linear.x) > max_speed_ ||
        std::abs(cmd_vel.twist.linear.y) > max_speed_ ||
        std::abs(cmd_vel.twist.angular.z) > max_angular_speed_)
    {
      RCLCPP_WARN(logger_, "Command velocity exceeds limits");
      return false;
    }

    // Check for collisions
    nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *> checker(costmap_ros_->getCostmap());
    geometry_msgs::msg::PoseStamped current_pose;
    current_pose.header = cmd_vel.header;
    current_pose.pose.position.x = 0.0;
    current_pose.pose.position.y = 0.0;
    double cost = checker.footprintCostAtPose(
        current_pose.pose.position.x,
        current_pose.pose.position.y,
        0.0,
        costmap_ros_->getRobotFootprint());
    if (cost >= safety_threshold_)
    {
      RCLCPP_WARN(logger_, "Command velocity would cause collision, cost: %f", cost);
      return false;
    }

    return true;
  }

  geometry_msgs::msg::TwistStamped SicnavController::smoothVelocity(
      const geometry_msgs::msg::TwistStamped &cmd_vel)
  {
    geometry_msgs::msg::TwistStamped smoothed_cmd = cmd_vel;
    if (!prev_cmd_vel_.header.frame_id.empty())
    {
      smoothed_cmd.twist.linear.x =
          (1.0 - smoothing_factor_) * cmd_vel.twist.linear.x +
          smoothing_factor_ * prev_cmd_vel_.twist.linear.x;
      smoothed_cmd.twist.linear.y =
          (1.0 - smoothing_factor_) * cmd_vel.twist.linear.y +
          smoothing_factor_ * prev_cmd_vel_.twist.linear.y;
      smoothed_cmd.twist.angular.z =
          (1.0 - smoothing_factor_) * cmd_vel.twist.angular.z +
          smoothing_factor_ * prev_cmd_vel_.twist.angular.z;
    }
    prev_cmd_vel_ = smoothed_cmd;
    return smoothed_cmd;
  }
} // namespace nav2py_sicnav_controller

PLUGINLIB_EXPORT_CLASS(nav2py_sicnav_controller::SicnavController, nav2_core::Controller);