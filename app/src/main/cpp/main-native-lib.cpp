#include <jni.h>
#include <android/log.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include "face-detection.h"

#define TAG "NativeLib"

using namespace std;
using namespace cv;

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
}