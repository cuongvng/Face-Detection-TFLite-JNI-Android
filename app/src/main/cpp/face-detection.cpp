#include "face-detection.h"

FaceDetector::FaceDetector(char* buffer, long size) {
    assert(mModelSize > 0);
    mModelSize = size;
    mModelBuffer = (char*) malloc(sizeof(char) * mModelSize);
    memcpy(mModelBuffer, buffer, sizeof(char) * mModelSize);
    loadModel();
}

void FaceDetector::loadModel() {

    mModel = tflite::FlatBufferModel::BuildFromBuffer(mModelBuffer, sizeof(char) * mModelSize);
    assert(mModel != nullptr);

    // Build the interpreter with the InterpreterBuilder.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*mModel, resolver);
    builder(&mInterpreter);
    assert(mInterpreter != nullptr);

    // Allocate tensor buffers.
    assert(mInterpreter->AllocateTensors() == kTfLiteOk);

    // TODO: set input and output tensors}

FaceDetector::~FaceDetector() {
    if(mModelBuffer != nullptr){
        free(mModelBuffer);
        mModelBuffer = nullptr;
    }
}