cmake_minimum_required(VERSION 3.10.2)

project("facedetection")

set(INCLUDE_DIRS
        ${OpenCV_DIR}/jni/include
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/tf-lite-api/include
        ${CMAKE_CURRENT_LIST_DIR}/tf-lite-api/include/flatbuffers/include)
include_directories(${INCLUDE_DIRS})

add_library( lib_opencv SHARED IMPORTED )
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION
        ${OpenCV_DIR}/libs/${ANDROID_ABI}/libopencv_java4.so)

add_library(tflite SHARED IMPORTED)
set_target_properties(tflite PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/tf-lite-api/generated-libs/${ANDROID_ABI}/libtensorflowlite.so)

add_library( native-lib SHARED main-native-lib.cpp face-detection.cpp)

find_library( log-lib  log ) # Library required by NDK
find_library( android-lib android ) # for AssetManager functionality

target_link_libraries( native-lib ${log-lib} ${android-lib} lib_opencv tflite)