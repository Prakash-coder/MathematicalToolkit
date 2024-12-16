
cmake_minimum_required(VERSION 3.16) #here, i am requiring the user to have a 
										#set version of cmake

set(CMAKE_CXX_STANDARD 17) 
#using cxx+17

set(CMAKE_CXX_STANDARD_REQUIRED ON)
#C++17 needs to be used

#project name
project("MathematicalToolkit" VERSION 1.0.1
	LANGUAGES CXX
	DESCRIPTION "MathematicalToolkit: AutomaticDifferentiation and Linear Algebra Library For C++."
	HOMEPAGE_URL "https://github.com/NisharArjyal/MathematicalToolkit")

#include(GNUInstallDirs)

#project ko name set gareko, version pani

#source files ko address euta single variable ma wrap gareko

add_library(${PROJECT_NAME} INTERFACE)
#AD naam gareko library banako, hunuparxa, link hunxa
target_include_directories(${PROJECT_NAME} INTERFACE src/)


#target_include_directories(
  #${PROJECT_NAME}
  #INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SRC_DIR}/src>
			#$<INSTALL_INTERFACE:src>
  #PRIVATE: 
			  #${CMAKE_CURRENT_SRC_DIR}/src)



add_subdirectory(test)

		#target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)



#install(TARGETS ${PROJECT_NAME}
        #EXPORT ${PROJECT_NAME}_Targets
        #ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        #LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        #RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})


#include(CMakePackageConfigHelpers)
#write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
                                 #VERSION ${PROJECT_VERSION}
                                 #COMPATIBILITY SameMajorVersion)



#configure_package_config_file(
  #"${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  #"${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  #INSTALL_DESTINATION
  #${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

#install(EXPORT ${PROJECT_NAME}_Targets
        #FILE ${PROJECT_NAME}Targets.cmake
        #NAMESPACE ${PROJECT_NAME}::
        #DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

#install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              #"${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        #DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

#install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/ DESTINATION include)





#library ko incldue 

#set(LIBRARY_NAME
#    MathematicalToolkit
#)
#target_include_directories(${LIBRARY_NAME} PRIVATE
#    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBRARY_NAME}>
#    $<INSTALL_INTERFACE:include/${LIBRARY_NAME}>
#)
#target_include_directories(${LIBRARY_NAME} PUBLIC
#    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#    $<INSTALL_INTERFACE:include>
#)

#add_subdirectory(test)


#add_executable(test test/tests.cpp)
#target_link_libraries(test PUBLIC AD)


