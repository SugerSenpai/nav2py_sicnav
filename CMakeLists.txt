cmake_minimum_required(VERSION 3.8)
project(nav2py_sicnav_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(ament_cmake_venv REQUIRED)
find_package(ament_cmake_venv_uv REQUIRED)
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

set(RVO2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/nav2py_sicnav_controller/Python-RVO2")

add_custom_command(
  OUTPUT ${RVO2_DIR}/.git
  COMMAND git submodule update --init --recursive
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/nav2py_sicnav_controller
  COMMENT "Initializing Python-RVO2 submodule"
)
add_custom_target(RVO2_SUBMODULE ALL DEPENDS ${RVO2_DIR}/.git)

uv_venv(PROJECTFILE pyproject.toml)

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/python_version_check
  COMMAND ${CMAKE_BINARY_DIR}/venv/bin/python --version > ${CMAKE_BINARY_DIR}/python_version.txt
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS VENV_CREATED
  COMMENT "Checking Python version in virtual environment"
)
add_custom_target(PYTHON_VERSION_CHECK ALL DEPENDS ${CMAKE_BINARY_DIR}/python_version_check)

venv_ensure_pip()

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/rvo2_installed
  COMMAND ${CMAKE_COMMAND} -E env CXX=/usr/bin/g++ ${CMAKE_BINARY_DIR}/venv/bin/pip install -e ${RVO2_DIR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS VENV_CREATED RVO2_SUBMODULE PYTHON_VERSION_CHECK
  COMMENT "Installing Python-RVO2 with g++"
)
add_custom_target(RVO2_INSTALL ALL DEPENDS ${CMAKE_BINARY_DIR}/rvo2_installed)

nav2py_package(nav2py_sicnav_controller)

include_directories(
  include
  ${nav2py_INCLUDE_DIRS}
)

add_library(nav2py_sicnav_controller SHARED
  src/sicnav_controller.cpp
)
target_compile_definitions(nav2py_sicnav_controller PUBLIC "PLUGINLIB__DISABLE_BOOST_FUNCTIONS")
ament_target_dependencies(nav2py_sicnav_controller
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

install(TARGETS nav2py_sicnav_controller
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/
  DESTINATION share/${PROJECT_NAME}/include/
)
install(DIRECTORY nav2py_sicnav_controller/
  DESTINATION share/${PROJECT_NAME}/nav2py_sicnav_controller/
)
install(DIRECTORY nav2py_sicnav_controller/safe-interactive-crowdnav/
  DESTINATION share/${PROJECT_NAME}/safe-interactive-crowdnav/
)
install(DIRECTORY nav2py_sicnav_controller/Python-RVO2/
  DESTINATION share/${PROJECT_NAME}/Python-RVO2/
)
install(FILES nav2py_sicnav_controller.xml
  DESTINATION share/${PROJECT_NAME}
)

pluginlib_export_plugin_description_file(nav2_core nav2py_sicnav_controller.xml)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_libraries(nav2py_sicnav_controller)
ament_export_dependencies(ament_cmake rclcpp rclcpp_lifecycle pluginlib tf2 tf2_ros tf2_geometry_msgs geometry_msgs nav_msgs nav2_costmap_2d nav2_core nav2_util nav2py sensor_msgs ament_index_cpp)

ament_package()