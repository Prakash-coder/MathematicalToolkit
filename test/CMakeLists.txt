cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
add_executable(performedtests ${FILES})
target_link_libraries(performedtests INTERFACE AD)
target_include_directories(performedtests INTERFACE $(PROJECT_SOURCE_DIR)/src)


#find_package(AD CONFIG REQUIRED)
#add_library(${PROJECT_NAME} AD)
#add_executable(${PROJECT_NAME} test.cpp)
#target_link_libraries(${PROJECT_NAME} AD::AD)




