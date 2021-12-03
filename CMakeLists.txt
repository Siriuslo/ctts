cmake_minimum_required(VERSION 3.0.2)
project( DisplayImage )

find_package( OpenCV 4 REQUIRED )
include_directories( 
    ${OpenCV_INCLUDE_DIRS}
    ${CURL_INCLUDE_DIR}
     )



add_executable( ctts 
src/ctts.cpp 
src/include/run_yolo.cpp)
target_link_libraries( ctts
 ${OpenCV_LIBS} 
 curl
 )

add_executable( test 
src/test.cpp 
src/include/run_yolo.cpp)
target_link_libraries( test
 ${OpenCV_LIBS} 
 curl
 )