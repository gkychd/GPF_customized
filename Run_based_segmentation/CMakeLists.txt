cmake_minimum_required(VERSION 2.8.3)
project(points_preprocessor_usi)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  pcl_ros
  pcl_conversions
  velodyne_pointcloud
)

catkin_package(CATKIN_DEPENDS
	sensor_msgs
	velodyne_pointcloud
)

###########
## Build ##
###########

include_directories(
	include
  ${catkin_INCLUDE_DIRS}
)
message(include)
find_package(Eigen3 REQUIRED)
find_package(Boost 1.54 REQUIRED)

#SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")

link_directories(${PCL_LIBRARY_DIRS})

#Ground Plane Fitter
#find_package(PCL 1.8 REQUIRED)
#add_definitions(${PCL_DEFINITIONS})

#add_executable(groundplanfit 
	#nodes/ground_filter/groundplanfit.cpp
#)

#target_include_directories(groundplanfit PRIVATE
	#${PCL_INCLUDE_DIRS}
#)

#target_link_libraries(groundplanfit
	#${catkin_LIBRARIES}
	#${PCL_LIBRARIES}
#)

#Scan Line Run
#add_executable(scanlinerun 
	#nodes/clustering/scanlinerun.cpp
#)

#target_include_directories(scanlinerun PRIVATE
	#${PCL_INCLUDE_DIRS}
#)

#target_link_libraries(scanlinerun
	#${catkin_LIBRARIES}
	#${PCL_LIBRARIES}
#)

#Ground Plane Fitter
find_package(PCL 1.8 REQUIRED)
add_definitions(${PCL_DEFINITIONS})

add_executable(offline_gs 
	nodes/ground_filter/offline_gs.cpp
)

target_include_directories(offline_gs PRIVATE
	${PCL_INCLUDE_DIRS}
)

target_link_libraries(offline_gs
	${catkin_LIBRARIES}
	${PCL_LIBRARIES}
)

