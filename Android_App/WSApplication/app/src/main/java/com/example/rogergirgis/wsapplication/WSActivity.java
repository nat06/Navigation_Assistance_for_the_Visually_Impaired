package com.example.rogergirgis.wsapplication;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.SystemClock;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.WindowManager;

import com.example.rogergirgis.wsapplication.OverlayView.DrawCallback;
import com.example.rogergirgis.wsapplication.env.BorderedText;
import com.example.rogergirgis.wsapplication.env.ImageUtils;
import com.example.rogergirgis.wsapplication.env.Logger;

import org.pielot.openal.SoundEnv;
import org.pielot.openal.Source;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Vector;

public class WSActivity extends CameraActivity implements OnImageAvailableListener, SensorEventListener {  //, SensorEventListener

    private static final Logger LOGGER = new Logger();

    protected static final boolean SAVE_PREVIEW_BITMAP = true;

    private ResultsView resultsView;

    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private long startTime;
    private long lastProcessingTimeMs;

    private BorderedText borderedText;
    private Integer sensorOrientation;
    private Classifier classifier;
    private Classifier unknownClassifier;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private static final boolean MAINTAIN_ASPECT = false;

    private static final Size DESIRED_PREVIEW_SIZE = new Size(480, 640);
    private static final float TEXT_SIZE_DIP = 10;

    private static final int INPUT_SIZE = 224;

    private static final String INPUT_NAME = "input_1";

    private static String OUTPUT_NAME;
    private static String MODEL_FILE;
    private static int IMAGE_MEAN;
    private static float IMAGE_STD;

//    private static final String OUTPUT_NAME_UNK = "loss/Softmax";
    private static final String OUTPUT_NAME_UNK = "act_softmax/Softmax";
//    private static final String MODEL_FILE_UNK = "file:///android_asset/Mobilenet_8.pb";
    private static final String MODEL_FILE_UNK = "file:///android_asset/run2.pb";

    private static final int IMAGE_MEAN_UNK = 1;
    private static final float IMAGE_STD_UNK = 0;
    private static final String LABEL_FILE_UNK = "file:///android_asset/classes.8.txt";
    private int unknownBool = 0;
    private final int UNKNOWN_THRESHOLD = 5;
    private int unknownCounter = 0;

    private static final String LABEL_FILE = "file:///android_asset/classes.8.txt";

    private static final int MODEL_TYPE = 0;  // 0 for classification, 1 for regression (??)
    int mPrediction = 0;
    float resultsFloat;

    // Sensor Params
//    private OrientationProvider orientationProvider;
//    private Quaternion quaternion = new Quaternion();
    float mInitX, mX;
    float mRotationSpeed;
    int gyroState = 0;
    int deviationAmount;
    private SensorManager mSensorManager;
    Handler handler = new Handler();
    int useModelCounter = 0;
    int totalCOUNT = 10;
    private final float[] mRotationVectorReading = new float[3];
    private final float[] mRotationMatrix = new float[9];
    private final float[] mOrientationAngles = new float[3];


    // Saving information
    public static final String FOLDER_KEY = "0";
    public String folderNumber;
    public static final int defaultFolderNumber = 0;
    int IMG_NUMBER = 0;
    List<Float> csv_data = new ArrayList<>();
    String TAG = "Hello";


    // Sound output params
    SoundEnv soundEnv;
    Source sound;
    Source unknownSound;
    private int lastPlayed = 2;
    private float xPos = 0.0F;
    private float yPos = 0.0F;
    static final float[][] POSITIONS = {
            {0.0F, 0.0F},
            {2.0F, 0.0F},
            {5.0F, 0.0F},
            {8.0F, 0.0F},
            {0.0F, 100000.0F},
            {-8.0F, 0.0F},
            {-5.0F, 0.0F},
            {-2.0F, 0.0F},
    };
    private TextToSpeech mTTS;
    private int mSpeaking;
    private int mSpeakIntro = 0;
    long startTimeGyro;
    private long gyroLastProcessingTimeMs;

    // Prediction Combination/Averaging/KalmanFilter
    ArrayList<Integer> history = new ArrayList<>();
    public HashMap<Integer, Float> actionVoting = new HashMap<>();
    // Kalman Filter for Predictions
    float xt_pred = 4.0F;
    float pt_pred = 0.8F;
    float K_pred = 0.0F;
    float Q_pred = 0.01F;
    float R_meas_pred = 0.8F;
    int kalmanUnknownCounter = 0;
    // Kalman Filter for Angles
    float xt = 0.0F;
    float pt = 0.5F;
    float K = 0.0F;
    float Q = 0.01F;
    float R_meas = 1.0F;

    public WSActivity() {
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        loadConfiguration();

        if (    MODEL_TYPE == 0 ){
//            OUTPUT_NAME = "loss/Softmax";
            OUTPUT_NAME = "act_softmax/Softmax";
//            OUTPUT_NAME = "dense_5";
//            MODEL_FILE = "file:///android_asset/Mobilenet_8.pb";
//            MODEL_FILE = "file:///android_asset/frozen_graph1.pb";
            MODEL_FILE = "file:///android_asset/run3.pb";
//            MODEL_FILE = "file:///android_asset/MobileNet_v2_GoogleStreetView.pb";
            IMAGE_MEAN = 1;
            IMAGE_STD = 0;
        } else if(MODEL_TYPE == 1){
            //OUTPUT_NAME = "re_lu_1/Relu";
            OUTPUT_NAME = "dense_5/Relu6";
            //MODEL_FILE = "file:///android_asset/mobilenetV1.pb";
//            MODEL_FILE = "file:///android_asset/MobileNet_v2_GoogleStreetView.pb";
//            MODEL_FILE = "file:///android_asset/frozen_graph1.pb";
            MODEL_FILE = "file:///android_asset/run3.pb";
            IMAGE_MEAN = 128;
            IMAGE_STD = 1;
        }

        this.soundEnv = SoundEnv.getInstance(this);
        try {
            this.soundEnv.setListenerOrientation(0);
            this.soundEnv.setListenerPos(0, 0, 0);

            this.sound = new Source(soundEnv.addBuffer("continual"));
            this.sound.setPosition(xPos, yPos, 0);

            this.unknownSound = new Source(soundEnv.addBuffer("unknown"));
            this.unknownSound.setPosition(xPos, yPos, 0);

        } catch (IOException e) {
            e.printStackTrace();
        }

        mTTS = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int i) {
                if (i == TextToSpeech.SUCCESS){
                    int anyError = mTTS.setLanguage(Locale.CANADA);

                    if (anyError == TextToSpeech.LANG_MISSING_DATA
                            || anyError == TextToSpeech.LANG_NOT_SUPPORTED){
                        LOGGER.i("TTS ", "Language Not supported");
                    }
                } else{
                    LOGGER.i("TTS ", "Initialization Failed");
                }
            }
        });
        mTTS.setSpeechRate(1.0F);

        mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);

        final String root = Environment.getExternalStorageDirectory().getAbsolutePath() +
                File.separator + "WalkingStraight" + File.separator + "Orientation" +
                File.separator + folderNumber;
        LOGGER.i(root);
        final File myDir = new File(root);

        if (myDir.isDirectory()) {
            String[] children = myDir.list();
            for (String aChildren : children) {
                new File(myDir, aChildren).delete();
            }
        }
        if (!myDir.mkdirs()) {
            LOGGER.i("Make dir failed");
        }

    }

    private void loadConfiguration() {
        Bundle configuration  = getIntent().getExtras();
        folderNumber = Integer.toString(configuration.getInt(FOLDER_KEY, defaultFolderNumber));
    }

    @Override
    public synchronized void onStop() {
        this.sound.stop();
        this.unknownSound.stop();

        if (mTTS != null) {
            mTTS.stop();
            mTTS.shutdown();
        }
        super.onStop();
    }

    @Override
    public void onStart(){
        super.onStart();
    }

    @Override
    public void onResume() {
        // Ideally a game should implement onResume() and onPause()
        // to take appropriate action when the activity looses focus

        Sensor rotationVector = mSensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR);
        if (rotationVector != null) {
            mSensorManager.registerListener(this, rotationVector,
                    SensorManager.SENSOR_DELAY_NORMAL, SensorManager.SENSOR_DELAY_UI);
        }

        Sensor gyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        if (gyroscope != null) {
            mSensorManager.registerListener(this, gyroscope,
                    SensorManager.SENSOR_DELAY_NORMAL, SensorManager.SENSOR_DELAY_UI);
        }

        super.onResume();
    }

    @Override
    public void onPause() {
        // Ideally a game should implement onResume() and onPause()
        // to take appropriate action when the activity looses focus
        super.onPause();
        save_csv();
        mSensorManager.unregisterListener(this);
    }

    @Override
    protected void processImage() {
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

        // For examining the actual TF input.
//        if (SAVE_PREVIEW_BITMAP) {
//            ImageUtils.saveBitmap(croppedBitmap, IMG_NUMBER);
//        }
//        runInBackground( new Runnable() {
        Runnable r = new Runnable() {
            @Override
            public void run() {
                if (mSpeakIntro == 0){
                    if (Integer.parseInt(folderNumber) == 0) {
                        speak("Rotate away from the auditory cues as slowly as possible, " +
                                "If no sound is produced, it means you are correctly oriented. ");
                        try {
                            Thread.sleep(500);
                        } catch (InterruptedException ex) {
                            android.util.Log.d("WSApplication",
                                    ex.toString());
                        }
                        //playTutorial();
                    }

                    speak("Please begin slowly rotating while I determine " +
                            "the correct heading.");
                    try { Thread.sleep(200); }
                    catch (InterruptedException ex) { android.util.Log.d("WSApplication",
                            ex.toString());}
                    mSpeakIntro++;
                    lastPlayed = 0;
                }

                List<Classifier.Recognition> results = null;
                List<Classifier.Recognition> unkResults = null;
                mPrediction = 0;
//                if (useModelCounter < totalCOUNT) {  // Use the model for predictions

                startTime = SystemClock.uptimeMillis();

                if (MODEL_TYPE == 0) {
                    results = classifier.recognizeImage(croppedBitmap, MODEL_TYPE);
                    resultsFloat = Float.parseFloat(results.get(0).getId());
                } else if (MODEL_TYPE == 1) {
                    unknownCounter++;
                    if (unknownBool == 1 || unknownCounter >= UNKNOWN_THRESHOLD) {
                        unkResults = unknownClassifier.recognizeImage(croppedBitmap, 0);
                        if (Integer.parseInt(unkResults.get(0).getId()) == 0) {
                            resultsFloat = 0.0F;
                            unknownBool = 1;
                        } else {
                            unknownBool = 0;
                        }
                        unknownCounter = 0;
                    }
                    if (unknownBool == 0) {
                        results = classifier.recognizeImage(croppedBitmap, 1);
                        resultsFloat = results.get(0).getConfidence() + 1.0F;
                        if (resultsFloat > 7.0F) {
                            resultsFloat = 7.0F;
                        }
                    }
                }

                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                LOGGER.i("Detect: %s", results);
                LOGGER.i("Total Detection Time: " + lastProcessingTimeMs);

                csv_data.add(resultsFloat);
                mPrediction = (int) applyKalmanFilterForPrediction(resultsFloat);
                LOGGER.i("Filtered Prediction: %s", mPrediction);
                csv_data.add((float) mPrediction);
                evaluateModelResult(mPrediction);
//                }
//                } else {  // Start using Gyroscope for position
//                    updateOrientationAngles();
//                    mX = applyKalmanFilterForAngle((float) (mOrientationAngles[0] * 180.0F / Math.PI));
//                    csv_data.add(mX);
//                    obtainPredictionFromAngles();
//                    csv_data.add((float) mPrediction);
//                    LOGGER.i("Prediction: %s", mPrediction);
//                }

                playSound(mPrediction);

                // Save IMU data in a csv file with the accompanying image
                cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                requestRender();
                readyForNextImage();
                ImageUtils.saveBitmap(croppedBitmap, IMG_NUMBER, folderNumber);
                IMG_NUMBER++;

                // Display model's result
                if (resultsView == null) {
                    resultsView = findViewById(R.id.results);
                }
                resultsView.setResults(results);
            }
        };
        handler.postDelayed(r, 800);
        //});
    }

//        Runnable r = new Runnable() {
//            public void run() {
//                requestRender();
//                readyForNextImage();
//                displayNewOrientation();
//                ImageUtils.saveBitmap(croppedBitmap, IMG_NUMBER);
//                //handler.postDelayed(this, 15000);
//                IMG_NUMBER++;
//                cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
//            }
//        };
//        handler.postDelayed(r, 400);

    private void obtainPredictionFromAngles() {
        LOGGER.i("mX: %f", mX);
        mPrediction = 4;

        if ((Math.abs(mX - mInitX) > 3.0) && (Math.abs(mX - mInitX) < 8.0)){
            deviationAmount = 1;
        } else if ((Math.abs(mX - mInitX) > 8.0) && (Math.abs(mX - mInitX) < 12.0)){
            deviationAmount = 2;
        } else if (Math.abs(mX - mInitX) > 12.0){
            deviationAmount = 3;
        } else {
            deviationAmount = 0;
        }

        if (mX - mInitX < 0.0) {
            mPrediction = mPrediction + deviationAmount;
        } else{
            mPrediction = mPrediction - deviationAmount;
        }
    }

    private void evaluateModelResult(int predictedValue){
        if (predictedValue == 4) {
//            useModelCounter++;
            useModelCounter = 0; // in order to use the model only for predictions
        } else {
            useModelCounter = 0;
        }
        // Start the sensor
        if (useModelCounter == totalCOUNT) {
            // Recording something in file to say we switched from model to IMU
            csv_data.add((float) -1000);

            updateOrientationAngles();
            mInitX = (float) (mOrientationAngles[0]*180.0F/Math.PI);
            LOGGER.i("mInitX: %f", mInitX);
            speak("Correct heading determined, when you know it is safe begin " +
                    "Crossing.");
        }
    }


    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_GAME_ROTATION_VECTOR) {
            System.arraycopy(event.values, 0, mRotationVectorReading,
                    0, mRotationVectorReading.length);
        } else if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            if (useModelCounter < totalCOUNT){  // only do when using the model
                mRotationSpeed = Math.abs((float) (event.values[0]*180F/Math.PI));
                if (mRotationSpeed > 30.0F && gyroState == 0) {
                    gyroState = 1;
                    startTimeGyro = SystemClock.uptimeMillis();
                    mTTS.speak("Rotate slower", TextToSpeech.QUEUE_FLUSH, null);
                }
                gyroLastProcessingTimeMs = SystemClock.uptimeMillis() - startTimeGyro;
                if (mRotationSpeed < 30.0F && gyroState == 1 && gyroLastProcessingTimeMs > 5){
                    gyroState = 0;
                }
            }
        }
    }


    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {
        // Do something here if sensor accuracy changes.
        // You must implement this callback in your code.
    }


    public void updateOrientationAngles() {
        // Update rotation matrix, which is needed to update orientation angles.
        SensorManager.getRotationMatrixFromVector(mRotationMatrix, mRotationVectorReading);
        // "mRotationMatrix" now has up-to-date information.
        SensorManager.getOrientation(mRotationMatrix, mOrientationAngles);
        // "mOrientationAngles" now has up-to-date information.

        // Add orientation Angles to csv file
        csv_data.add((float) (mOrientationAngles[0]*180.0F/Math.PI));
//        csv_data.add((float) (mOrientationAngles[1]*180.0F/Math.PI));
//        csv_data.add((float) (mOrientationAngles[2]*180.0F/Math.PI));
    }


    private void save_csv() {

        final String fname = "csv_data.csv";
        final String root = Environment.getExternalStorageDirectory().getAbsolutePath() +
                File.separator + "WalkingStraight" + File.separator + "Orientation" +
                File.separator + folderNumber;
        final File myDir = new File(root);

        final File file = new File(myDir, fname);
        if (file.exists()) {
            file.delete();
        }
        try {
            // Make sure the Pictures directory exists.
            file.createNewFile();
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            FileOutputStream outputStream = new FileOutputStream(file, true);
            for (int i = 0; i < csv_data.size(); i += 5) {
                outputStream.write((csv_data.get(i) + ",").getBytes());  // X
                outputStream.write((csv_data.get(i + 1) + ",").getBytes());  // Y
                outputStream.write((csv_data.get(i + 2) + ",").getBytes());  // Z
                outputStream.write((csv_data.get(i + 3) + ",").getBytes());  // X_kalman
                outputStream.write((csv_data.get(i + 4) + "\n").getBytes());  // Prediction
            }
            outputStream.close();
        } catch (Exception e) {
            Log.e("Exception", "File write failed: " + e.toString());
        }
    }


    public int keepHistory(String predictedValue) {
        int predictedInt = Integer.parseInt(predictedValue);

        // Discount all existing entries
        for(int i=0; i < 8; i++) {
            if(actionVoting.containsKey(i)) {
                actionVoting.put(i, actionVoting.get(i)*0.65F);
            }
        }

        // Put the new value of the predicted action
        if (!actionVoting.containsKey(predictedInt))
            actionVoting.put(predictedInt, 1.0F);
        else {
            actionVoting.put(predictedInt, actionVoting.get(predictedInt) + 1.0F);
        }
        LOGGER.i("HashMap Values: %s", actionVoting);

        // Decide on the most voted action
        int mostVoted = 0;
        for(int i = 1; i < 8; i++) {
            if (actionVoting.containsKey(i)) {
                if (actionVoting.get(i) > actionVoting.get(mostVoted)) {
                    mostVoted = i;
                }
            }
        }
        return mostVoted;
    }


    public float applyKalmanFilterForPrediction(Float predictedValue) {
        if (predictedValue != 0.0F) {
            pt_pred = pt_pred + Q_pred;
            K_pred = pt_pred / (pt_pred + R_meas_pred);
            xt_pred = xt_pred + K_pred * (predictedValue - xt_pred);
            pt_pred = pt_pred * (1.0F - K_pred);
            LOGGER.i("Kalman Filter for Prediction: Gain = %s, ErrorEstimate = %s, Estimate = %s, " +
                    "Measurement = %s.", K_pred, pt_pred, xt_pred, predictedValue);
            kalmanUnknownCounter = 0;
        } else{
            kalmanUnknownCounter++;
        }

        LOGGER.i("Kalman Filter for Prediction: Rounded Prediction = %s", Math.round(xt_pred));
        LOGGER.i("Kalman Filter for Prediction: UnknownCounter = %s", kalmanUnknownCounter);
        if (kalmanUnknownCounter >= 3) {
            LOGGER.i("Kalman Filter for Prediction: Here, returning = %s", 0);
            return 0;
        }

        return Math.round(xt_pred);
    }


    public float applyKalmanFilterForAngle(float angle) {

        pt = pt + Q;
        K = pt / (pt + R_meas);
        xt = xt + K*(angle - xt);
        pt = pt*(1.0F - K);
        LOGGER.i("Kalman Filter for Angle: Gain = %s, ErrorEstimate = %s, Estimate = %s, " +
                "Measurement = %s.", K, pt, xt, angle);
        return xt;
    }


    protected void speak(String speechString){
        this.sound.stop();
        this.unknownSound.stop();
        mTTS.speak(speechString, TextToSpeech.QUEUE_FLUSH, null);
        while (mTTS.isSpeaking()) {
            // do nothing
        }

        if (lastPlayed == 1)        this.sound.play(true);
        else if (lastPlayed == 0)        this.unknownSound.play(true);
    }

    protected void playTutorial(){
        WSActivity.this.sound.setPosition(-2.0F, 0, 0);
        this.sound.play(true);
        try { Thread.sleep(2000); }
        catch (InterruptedException ex) { android.util.Log.d("WSApplication", ex.toString());}
        this.sound.stop();
        mTTS.speak("indicates you should rotate right.", TextToSpeech.QUEUE_FLUSH, null);
        while (mTTS.isSpeaking()) {
            // do nothing
        }

        WSActivity.this.sound.setPosition(2.0F, 0, 0);
        this.sound.play(true);
        try { Thread.sleep(2000); }
        catch (InterruptedException ex) { android.util.Log.d("WSApplication", ex.toString());}
        this.sound.stop();
        mTTS.speak("indicates you should rotate left.", TextToSpeech.QUEUE_FLUSH, null);
        while (mTTS.isSpeaking()) {
            // do nothing
        }

        WSActivity.this.unknownSound.setPosition(0, 0, 0);
        this.unknownSound.play(true);
        try { Thread.sleep(2000); }
        catch (InterruptedException ex) { android.util.Log.d("WSApplication", ex.toString());}
        this.unknownSound.stop();
        mTTS.speak("indicates something is obstructing the view.",
                TextToSpeech.QUEUE_FLUSH, null);
        while (mTTS.isSpeaking()) {
            // do nothing
        }

        try { Thread.sleep(1000); }
        catch (InterruptedException ex) { android.util.Log.d("WSApplication", ex.toString());}
    }


    protected void playSound(int predictedInt) {
        xPos = POSITIONS[predictedInt][0];
        yPos = POSITIONS[predictedInt][1];
        WSActivity.this.sound.setPosition(xPos, yPos, 0);
        WSActivity.this.unknownSound.setPosition(xPos, yPos, 0);
        if(predictedInt == 0) {
            if(lastPlayed == 1) {
                this.sound.stop();
                this.unknownSound.play(true);
            }
            lastPlayed = 0;
        } else {
            if(lastPlayed == 0) {
                this.sound.play(true);
                this.unknownSound.stop();
            }
            lastPlayed = 1;
        }
    }


    @Override
    protected void onPreviewSizeChosen(Size size, int rotation) {
        final float textSizePx = TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        classifier =
                TensorFlowImageClassifier.create(
                        getAssets(),
                        MODEL_FILE,
                        LABEL_FILE,
                        INPUT_SIZE,
                        IMAGE_MEAN,
                        IMAGE_STD,
                        INPUT_NAME,
                        OUTPUT_NAME);

        if (MODEL_TYPE == 1) {
            unknownClassifier =
                    TensorFlowImageClassifier.create(
                            getAssets(),
                            MODEL_FILE_UNK,
                            LABEL_FILE_UNK,
                            INPUT_SIZE,
                            IMAGE_MEAN_UNK,
                            IMAGE_STD_UNK,
                            INPUT_NAME,
                            OUTPUT_NAME_UNK);
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = 180; // rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Config.ARGB_8888);  //INPUT_SIZE

        frameToCropTransform = ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                INPUT_SIZE, INPUT_SIZE, // INPUT_SIZE
                sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        renderDebug(canvas);
                    }
                });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    private void renderDebug(final Canvas canvas) {
        if (!isDebug()) {
            return;
        }
        final Bitmap copy = cropCopyBitmap;
        if (copy != null) {
            final Matrix matrix = new Matrix();
            final float scaleFactor = 2;
            matrix.postScale(scaleFactor, scaleFactor);
            matrix.postTranslate(
                    canvas.getWidth() - copy.getWidth() * scaleFactor,
                    canvas.getHeight() - copy.getHeight() * scaleFactor);
            canvas.drawBitmap(copy, matrix, new Paint());

            final Vector<String> lines = new Vector<String>();
            if (classifier != null) {
                String statString = classifier.getStatString();
                String[] statLines = statString.split("\n");
                for (String line : statLines) {
                    lines.add(line);
                }
            }

            lines.add("Frame: " + previewWidth + "x" + previewHeight);
            lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
            lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
            lines.add("Rotation: " + sensorOrientation);
            lines.add("Inference time: " + lastProcessingTimeMs + "ms");
            lines.add("Aggr. Action: " + mPrediction);

            borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
        }
    }

    @Override
    public void onSetDebug(boolean debug) {
        classifier.enableStatLogging(debug);
    }

}
