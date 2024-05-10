cmake_minimum_required(VERSION 3.10)

# Set a default C++ compiler if one wasn't provided
set(CMAKE_CXX_COMPILER "icx")
if(NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "g++")
endif()

# set(CMAKE_CXX_FLAGS "-O3 -std=c++17")
set(CMAKE_CXX_FLAGS "-O2 -g -std=c++17")
if(CMAKE_CXX_COMPILER STREQUAL "icx")      # == 如果是intel编译器
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-zmm-usage=high -xHost")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost -mavx512f -Wno-c++17-extensions -qopenmp")

  # 对于Intel编译器 使用 KMP_AFFINITY 设置线程亲和
  add_definitions(-DKMP_AFFINITY=compact,1,0,granularity=fine)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fopenmp")
endif()

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBSR_ANA ")

# Set the project name and version
project(LeSpMV VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify the C++ standard
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# make a directory named features for save matrix features
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/features)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/performance)

# Include directories for the library
include_directories(include)

# Gather the library sources
file(GLOB LIB_SOURCES "src/*.cpp")
file(GLOB LIB_UTILS   "utils/*.cpp")

# Create the static library
add_library(LeSPMV_static STATIC ${LIB_SOURCES} ${LIB_UTILS})

# Create the shared library
add_library(LeSPMV_shared SHARED ${LIB_SOURCES} ${LIB_UTILS})

# It's common practice to output libraries with the same name, with different extensions.
# CMake automatically appends the appropriate extension for static (.a) and shared (.so or .dll) libraries.
set_target_properties(LeSPMV_shared PROPERTIES OUTPUT_NAME LeSPMV)
set_target_properties(LeSPMV_static PROPERTIES OUTPUT_NAME LeSPMV)

# 定义源脚本目录和目标脚本目录变量
set(SOURCE_SCRIPT_DIR ${CMAKE_SOURCE_DIR}/script)
set(DESTINATION_SCRIPT_DIR ${CMAKE_BINARY_DIR})

add_custom_command(TARGET LeSPMV_static POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                           ${SOURCE_SCRIPT_DIR} ${DESTINATION_SCRIPT_DIR}
                   COMMENT "Copying scripts to build directory")

# Gather the test sources
file(GLOB TEST_SOURCES "test/*.cpp")

# Allow the test executables to find the library headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# Create an executable for each test source
foreach(test_src ${TEST_SOURCES})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  # Add compiler options here if they are specific to the test executables
  target_compile_options(${test_name} PRIVATE -O3 -Wall)

  # target_link_libraries(${test_name} LeSPMV_shared stdc++)
  
  # Check if the current source file is the baseline MKL test
  if(${test_name} STREQUAL "baseline_mkl_csr")
    # Link with MKL. The method to link may depend on your setup.
    # This example assumes you're using the Intel Compiler or you have MKL setup to integrate with CMake.
    find_package(MKL CONFIG REQUIRED)
    target_link_libraries(${test_name} MKL::MKL LeSPMV_shared stdc++)
  else()
    target_link_libraries(${test_name} LeSPMV_shared stdc++)
  endif()

endforeach(test_src)

# Message to print out the compiler being used
message(STATUS "Using CXX compiler: ${CMAKE_CXX_COMPILER}")
# Also directly access the CMAKE_CXX_FLAGS variable
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")