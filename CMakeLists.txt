cmake_minimum_required(VERSION 3.8)
project(nav2py_sicnav_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(pluginlib REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(nav2_common REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav2py REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(ament_cmake_venv REQUIRED)
find_package(ament_cmake_venv_uv REQUIRED)

uv_venv(PROJECTFILE pyproject.toml)
nav2py_package(nav2py_sicnav_controller)

include_directories(
  include
  ${nav2py_INCLUDE_DIRS}
)

set(dependencies
  rclcpp
  rclcpp_lifecycle
  pluginlib
  tf2
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
  nav_msgs
  nav2_costmap_2d
  nav2_core
  nav2_util
  nav2py
  sensor_msgs
  ament_index_cpp
)

add_library(nav2py_sicnav_controller SHARED
  src/sicnav_controller.cpp)

target_include_directories(nav2py_sicnav_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_definitions(nav2py_sicnav_controller PUBLIC "PLUGINLIB__DISABLE_BOOST_FUNCTIONS")

ament_target_dependencies(nav2py_sicnav_controller
  ${dependencies}
)

set(SETUP_RVO2_PY "${CMAKE_CURRENT_SOURCE_DIR}/setup_rvo2.py")
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/rvo2_setup
  COMMAND ${Python3_EXECUTABLE} ${SETUP_RVO2_PY}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml ${SETUP_RVO2_PY}
  COMMENT "Setting up Python-RVO2 dependency"
)

add_custom_target(RVO2_SETUP ALL DEPENDS ${CMAKE_BINARY_DIR}/rvo2_setup)

install(TARGETS nav2py_sicnav_controller
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include/
)

install(FILES nav2py_sicnav_controller.xml
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_libraries(nav2py_sicnav_controller)
ament_export_dependencies(${dependencies})

pluginlib_export_plugin_description_file(nav2_core nav2py_sicnav_controller.xml)

ament_package()