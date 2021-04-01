package cuongvng.facedetection;

import android.Manifest;
import android.graphics.*;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.os.SystemClock;
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
    private final int maxFaces = 1;

    private int frameWidth = 0;
    private int frameHeight = 0;
    private int rotationToUser = 0;
    private Paint _paint = new Paint();
    private SurfaceView surfaceView;
    private CameraView cameraView;
    private boolean surfaceLocked = false;
    private Canvas canvas = null;
    private Path path = new Path();

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
        final long start = SystemClock.elapsedRealtime();
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
                maxFaces,
                frameWidth,
                frameHeight,
                rotationToUser
                );

        long elapsed = SystemClock.elapsedRealtime() - start;
        double fps = 1000.0/elapsed;
        Log.i(TAG, String.format("FPS: %f", fps));

        if (!surfaceLocked){
            canvas = surfaceView.getHolder().lockCanvas();
            if (detections.length==0){ // erase boxes
                canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.MULTIPLY);
                surfaceView.getHolder().unlockCanvasAndPost(canvas);
            }
            else {
                surfaceLocked = true;
                path.rewind();
            }
        }

        if (canvas != null && surfaceLocked) {
            canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.MULTIPLY);
            // Draw the detections, // TODO: limit number of detections
            for (byte i=0; i<maxFaces; i++)
                this.drawDetection(canvas, path, rotationToUser, detections, i);
            surfaceView.getHolder().unlockCanvasAndPost(canvas);
            surfaceLocked = false;
        }
    }

    private void drawDetection(Canvas canvas, Path p, int rotation, float[] detections, int idx){
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

        p.moveTo(x1, y1);
        p.lineTo(x1, y2);
        p.lineTo(x2, y2);
        p.lineTo(x2, y1);
        p.lineTo(x1, y1);

        canvas.drawPath(p, _paint);
    }
    private native long loadDetectorJNI(AssetManager assetManager, String filename);
    private native float[] detectJNI(long detectorPtr, byte[] src,
                                     float heatmapThreshold, float nmsThreshold, int maxFaces,
                                     int width, int heith, int rotation);
}
