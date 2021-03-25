//
// Created by Cuong V. Nguyen on 3/25/21.
//

#ifndef FACEDETECTION_FACE_DETECTION_H
#define FACEDETECTION_FACE_DETECTION_H

#include <cstdio>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

class FaceDetector{
private:
    char* mModelBuffer = nullptr;
    long mModelSize;
    std::unique_ptr<tflite::FlatBufferModel> mModel;
    std::unique_ptr<tflite::Interpreter> mInterpreter;

public:
    FaceDetector(char* buffer, long size);
    ~FaceDetector();
    void loadModel();
};

#endif //FACEDETECTION_FACE_DETECTION_H
