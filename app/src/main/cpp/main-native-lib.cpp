#include <jni.h>
#include <android/log.h>
#include <android/bitmap.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include "face-detection.h"

#define TAG "NativeLib"

void rotateMat(cv::Mat &matImage, int rotation);

extern "C" {
jlong JNICALL Java_cuongvng_facedetection_MainActivity_loadDetectorJNI(
        JNIEnv *env,
        jobject instance,
        jobject assetManager,
        jstring filename) {

    char* buffer = nullptr;
    long size = 0;
    const char* modelpath = env->GetStringUTFChars(filename, 0);

    if (!(env->IsSameObject(assetManager, NULL))) {
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    AAsset *asset = AAssetManager_open(mgr, modelpath, AASSET_MODE_UNKNOWN);
    assert(asset != nullptr);

    size = AAsset_getLength(asset);
    buffer = (char *) malloc(sizeof(char) * size);
    AAsset_read(asset, buffer, size);
    AAsset_close(asset);
    }

    jlong detectorPointer = (jlong) new FaceDetector(buffer, size, false);
    free(buffer);
    return detectorPointer;
}

jfloatArray JNICALL Java_cuongvng_facedetection_MainActivity_detectJNI(
        JNIEnv *env,
        jobject instance,
        jlong detectorPtr,
        jbyteArray src,
        jfloat heatmapThreshold,
        jfloat nmsThreshold,
        jint maxFaces,
        jint width, jint height, jint rotation) {

    // Frame bytes to Mat
    jbyte *yuv = env->GetByteArrayElements(src, 0);
    cv::Mat my_yuv(height + height / 2, width, CV_8UC1, yuv);
    cv::Mat frame(height, width, CV_8UC4);

    cv::cvtColor(my_yuv, frame, cv::COLOR_YUV2BGRA_NV21);

    rotateMat(frame, rotation);
    env->ReleaseByteArrayElements(src, yuv, 0);

    FaceDetector* detector = (FaceDetector*) detectorPtr;

    std::vector<FaceInfo> faces;
    detector->detect(frame, faces, heatmapThreshold, nmsThreshold, maxFaces);

    int resLen = faces.size()*N_FACE_FEATURES;
    jfloat jfaces[resLen];
    for (int i=0; i<faces.size(); i++){
        jfaces[i * N_FACE_FEATURES] = faces[i].x1;
        jfaces[i * N_FACE_FEATURES + 1] = faces[i].y1;
        jfaces[i * N_FACE_FEATURES + 2] = faces[i].x2;
        jfaces[i * N_FACE_FEATURES + 3] = faces[i].y2;
        jfaces[i * N_FACE_FEATURES + 4] = faces[i].score;
    }

    jfloatArray detections = env->NewFloatArray(resLen);
    env->SetFloatArrayRegion(detections, 0, resLen, jfaces);

    return detections;
}
}

void rotateMat(cv::Mat &matImage, int rotation) {
    if (rotation == 90) {
        transpose(matImage, matImage);
        flip(matImage, matImage, 1); //transpose+flip(1)=CW
    } else if (rotation == 270) {
        transpose(matImage, matImage);
        flip(matImage, matImage, 0); //transpose+flip(0)=CCW
    } else if (rotation == 180) {
        flip(matImage, matImage, -1);    //flip(-1)=180
    }
}
