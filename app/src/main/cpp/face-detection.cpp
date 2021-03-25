#include "face-detection.h"

FaceDetector::FaceDetector(char* buffer, long size) {
    assert(modelSize>0);
    modelSize = size;
    modelBuffer = (char*) malloc(sizeof(char) * modelSize);
    memcpy(modelBuffer, buffer, sizeof(char) * modelSize);
    loadModel();
}

void FaceDetector::loadModel() {

    model = tflite::FlatBufferModel::BuildFromBuffer(modelBuffer, sizeof(char) * modelSize);
    assert(model != nullptr);

    // Build the interpreter with the InterpreterBuilder.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    assert(interpreter != nullptr);

    // Allocate tensor buffers.
    assert(interpreter->AllocateTensors() == kTfLiteOk);

    // TODO: set input and output tensors
}

FaceDetector::~FaceDetector() {
    if(modelBuffer != nullptr){
        free(modelBuffer);
        modelBuffer = nullptr;
    }
}