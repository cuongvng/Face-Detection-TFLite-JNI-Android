package cuongvng.facedetection;

import android.Manifest;
import android.graphics.*;
import android.content.res.AssetManager;
import android.os.Bundle;
import androidx.core.app.ActivityCompat;
import androidx.appcompat.app.AppCompatActivity;

import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;

import com.otaliastudios.cameraview.CameraView;
import com.otaliastudios.cameraview.frame.Frame;

public class MainActivity extends AppCompatActivity{
    private static final String TAG = "MainActivity";
    private static final int CAMERA_PERMISSION_REQUEST = 1;

    private final String MODEL_FILE = "centerface_w640_h480.tflite";
    private long detectorPointer = 0L;
    private float heatmapThreshold = (float) 0.5;
    private float nmsThreshold = (float) 0.3;
    private final int nFaceInfo = 5;

    private int frameWidth = 0;
    private int frameHeight = 0;
    private int rotationToUser = 0;
    private Paint _paint = new Paint();
    private SurfaceView surfaceView;
    private CameraView cameraView;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);

        System.loadLibrary("native-lib");
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
        cameraView = this.findViewById(R.id.camera);
        surfaceView = this.findViewById(R.id.surfaceView);

        cameraView.setLifecycleOwner(this);
        cameraView.addFrameProcessor(frame -> {detectFaceNative(frame);});

        // init the paint for drawing the detections
        _paint.setColor(Color.RED);
        _paint.setStyle(Paint.Style.STROKE);
        _paint.setStrokeWidth(10f);

        // Set the detections drawings surface transparent
        surfaceView.setZOrderOnTop(true);
        surfaceView.getHolder().setFormat(PixelFormat.TRANSPARENT);

    }

    private void detectFaceNative(Frame frame){
        final long start = System.currentTimeMillis();
        if (detectorPointer == 0L){
            detectorPointer = loadDetectorJNI(this.getAssets(), MODEL_FILE);
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

        Canvas canvas = surfaceView.getHolder().lockCanvas();
        if (canvas != null) {
            canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.MULTIPLY);
            // Draw the detections, // TODO: limit number of detections
            this.drawDetection(canvas, rotationToUser, detections, 0);
            this.drawDetection(canvas, rotationToUser, detections, 1);
            this.drawDetection(canvas, rotationToUser, detections, 2);
            surfaceView.getHolder().unlockCanvasAndPost(canvas);
        }
    }

    private void drawDetection(Canvas canvas, int rotation, float[] detections, int idx){
        // Frame dimensions
        int w;
        int h;
        if (rotation == 0 || rotation == 180){
            w = frameWidth;
            h = frameHeight;
        }
        else{
            w = frameHeight;
            h = frameWidth;
        }

        float scaleX =  (float) cameraView.getWidth() / w;
        float scaleY = (float) cameraView.getHeight() / h;
        float xOffset =  (float) cameraView.getLeft();
        float yOffset = (float) cameraView.getTop();

        float x1 = xOffset + detections[idx*nFaceInfo] * scaleX;
        float y1 = yOffset + detections[idx*nFaceInfo + 1] * scaleY;
        float x2 = xOffset + detections[idx*nFaceInfo + 2] * scaleX;
        float y2 = yOffset + detections[idx*nFaceInfo + 3] * scaleY;

        Path p = new Path();
        p.moveTo(x1, y1);
        p.lineTo(x1, y2);
        p.lineTo(x2, y2);
        p.lineTo(x2, y1);
        p.lineTo(x1, y1);

        canvas.drawPath(p, _paint);
    }
    private native long loadDetectorJNI(AssetManager assetManager, String filename);
    private native float[] detectJNI(long detectorPtr, byte[] src,
                                     float heatmapThreshold, float nmsThreshold,
                                     int width, int heith, int rotation);
}
