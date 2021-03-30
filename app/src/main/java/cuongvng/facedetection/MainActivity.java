package cuongvng.facedetection;

import android.Manifest;
import android.app.Activity;
import android.graphics.*;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.os.Bundle;
import androidx.core.app.ActivityCompat;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;

import com.google.flatbuffers.kotlin.ByteArrayKt;
import com.otaliastudios.cameraview.CameraView;
import com.otaliastudios.cameraview.frame.Frame;

public class MainActivity extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "MainActivity";
    private static final int CAMERA_PERMISSION_REQUEST = 1;

    private final String MODEL_FILE = "centerface_w640_h480.tflite";
    private long detectorPointer = 0L;
    private float heatmapThreshold = 0.5;
    private float nmsThreshold = 0.3;
    private final int nFaceInfo = 5;

    private int frameWidth = 0;
    private int frameHeight = 0;
    private int rotationToUser = 0;
    private final Paint _paint = Paint();
//    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS) {
                Log.i(TAG, "OpenCV loaded successfully");

                // Load native library after(!) OpenCV initialization
                System.loadLibrary("native-lib");

//                mOpenCvCameraView.enableView();
            } else {
                super.onManagerConnected(status);
            }
        }
    };

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        // Permissions for Android 6+
        ActivityCompat.requestPermissions(
                this,
                new String[]{Manifest.permission.CAMERA},
                CAMERA_PERMISSION_REQUEST
        );
        // Load model
        detectorPointer = loadDetectorJNI(this.getAssets(), MODEL_FILE);

        setContentView(R.layout.activity_main);
        CameraView cameraView = this.findViewById<CameraView>(R.id.camera);
        cameraView.setLifecycleOwner(this);
        cameraView.addFrameProcessor(frame -> {detectFaceNative(frame);});

//        // init the paint for drawing the detections
//        _paint.color = Color.RED;
//        _paint.style = Paint.Style.STROKE;
//        _paint.strokeWidth = 3f;
//        _paint.textSize = 50f;
//        _paint.textAlign = Paint.Align.LEFT;
//
//        // Set the detections drawings surface transparent
//        surfaceView.setZOrderOnTop(true);
//        surfaceView.holder.setFormat(PixelFormat.TRANSPARENT);
//
    }

//    @Override
//    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
//        if (requestCode == CAMERA_PERMISSION_REQUEST) {
//            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
//                mOpenCvCameraView.setCameraPermissionGranted();
//            } else {
//                String message = "Camera permission was not granted";
//                Log.e(TAG, message);
//                Toast.makeText(this, message, Toast.LENGTH_LONG).show();
//            }
//        } else {
//            Log.e(TAG, "Unexpected permission request");
//        }
//    }

//    @Override
//    public void onPause() {
//        super.onPause();
//        if (mOpenCvCameraView != null)
//            mOpenCvCameraView.disableView();
//    }

//    @Override
//    public void onResume() {
//        super.onResume();
//        if (!OpenCVLoader.initDebug()) {
//            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
//            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
//        } else {
//            Log.d(TAG, "OpenCV library found inside package. Using it!");
//            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
//        }
//    }

    private void detectFaceNative(Frame frame){
        final long start = System.currentTimeMillis();
        if (detectorPointer == 0L){
            detectorPointer == loadDetectorJNI(this.getAssets(), MODEL_FILE);
        }

        frameHeight = frame.getSize().getHeight();
        frameWidth = frame.getSize().getWidth();
        rotationToUser = frame.getRotationToUser();

        float[] detections = this.detectJNI(
                detectorPointer,
                frame.getData(),
                heatmapThreshold,
                nmsThreshold,
                frameWidth,
                frameHeight,
                rotationToUser
                );

        final long elapsed = System.currentTimeMillis() - start;
        Log.i(TAG, "Detection elapsed time: ${elapsed}ms");

//        val canvas = surfaceView.holder.lockCanvas();
//        if (canvas != null) {
//            canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.MULTIPLY)
//            // Draw the detections, in our case there are only 3
//            this.drawDetection(canvas, rotationToUser, detections, 0);
//            this.drawDetection(canvas, rotationToUser, detections, 1);
//            this.drawDetection(canvas, rotationToUser, detections, 2);
//            surfaceView.holder.unlockCanvasAndPost(canvas);
//        }
    }
//    @Override
//    public void onDestroy() {
//        super.onDestroy();
//        if (mOpenCvCameraView != null)
//            mOpenCvCameraView.disableView();
//    }

//    @Override
//    public void onCameraViewStarted(int width, int height) {
//    }
//
//    @Override
//    public void onCameraViewStopped() {
//    }

//    @Override
//    public Mat onCameraFrame(CvCameraViewFrame frame) {
//        // get current camera frame as OpenCV Mat object
//        Mat mat = frame.rgba();
//
////        // native call to process current camera frame
////        adaptiveThresholdFromJNI(mat.getNativeObjAddr());
//
//        // return processed frame for live preview
//        return mat;
//    }

//    private void drawDetection(Canvas canvas, int rotation, float[] detections, int faceIndex){
//        // Frame dimensions
//        int w;
//        int h;
//        if (rotation == 0 || rotation == 180){
//            w = frameWidth;
//            h = frameHeight;
//        }
//        else{
//            w = frameHeight;
//            h = frameWidth;
//        }
//
//        float scaleX =  / w;
//
//
//    }
    private native long loadDetectorJNI(AssetManager assetManager, String filename);
    private native float[] detectJNI(long detectorPtr, ByteArrayKt src,
                                     float heatmapThreshold, float nmsThreshold,
                                     int width, int heith, int rotation);
}
