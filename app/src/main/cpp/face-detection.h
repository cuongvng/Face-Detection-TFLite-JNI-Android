//
// Created by Cuong V. Nguyen on 3/25/21.
//

#ifndef FACEDETECTION_FACE_DETECTION_H
#define FACEDETECTION_FACE_DETECTION_H

#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

enum{N_FACE_FEATURES=5}; // Number of element in struct FaceInfo

struct FaceInfo{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
};

class FaceDetector{
private:
    char* mModelBuffer = nullptr;
    long mModelSize;
    bool mModelQuantized = false;
    const int OUTPUT_WIDTH = 640;
    const int OUTPUT_HEIGHT = 480;
    std::unique_ptr<tflite::FlatBufferModel> mModel;
    std::unique_ptr<tflite::Interpreter> mInterpreter;
    TfLiteTensor* mInputTensor = nullptr;
    TfLiteTensor* mOutputHeatmap = nullptr;
    TfLiteTensor* mOutputScale = nullptr;
    TfLiteTensor* mOutputOffset = nullptr;

    int d_h;
    int d_w;
    float d_scale_h;
    float d_scale_w;
    float scale_w ;
    float scale_h ;
    int image_h;
    int image_w;

public:
    FaceDetector(char* buffer, long size, bool quantized=false);
    ~FaceDetector();
    void detect(cv::Mat img, std::vector<FaceInfo>& faces,
            float scoreThresh, float nmsThresh, int maxFaces);

private:
    void loadModel();
    void dynamic_scale(float in_w, float in_h);
    void postProcess(float* heatmap, float* scale, float* offset,
                     std::vector<FaceInfo>& faces,
                     float heatmapThreshold, float nmsThreshold, int maxFaces);
    void nms(std::vector<FaceInfo>& input, std::vector<FaceInfo>& output,
            float nmsThreshold, int maxFaces);
    std::vector<int> filterHeatmap(float* heatmap, int h, int w, float thresh);
    void getBox(std::vector<FaceInfo>& faces);
};

#endif //FACEDETECTION_FACE_DETECTION_H
