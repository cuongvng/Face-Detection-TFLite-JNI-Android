cmake_minimum_required(VERSION 3.10.2)

project("facedetection")

include_directories(${OpenCV_DIR}/jni/include)
add_library( lib_opencv SHARED IMPORTED )
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ${OpenCV_DIR}/libs/${ANDROID_ABI}/libopencv_java4.so)

set(INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/tf-lite-api/include
        ${CMAKE_CURRENT_LIST_DIR}/tf-lite-api/include/flatbuffers/include)

include_directories(${INCLUDE_DIRS})

link_directories(${CMAKE_CURRENT_LIST_DIR}/tf-lite-api/generated-libs/${ANDROID_ABI}/)

add_library( native-lib SHARED main-native-lib.cpp face-detection.cpp)
#target_include_directories( native-lib PUBLIC face-detection.h)

find_library( log-lib  log ) # Library required by NDK
find_library( android-lib android ) # for AssetManager functionality

target_link_libraries( native-lib ${log-lib} ${android-lib} lib_opencv libtensorflowlite.so)