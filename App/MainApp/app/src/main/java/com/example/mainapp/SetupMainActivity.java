package com.example.mainapp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Environment;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import java.io.File;
import java.io.IOException;

public class SetupMainActivity extends AppCompatActivity
{

    private Button recordButton;
    private MediaRecorder mediaRecorder;
    private File outputFile;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_setup_main);

        recordButton = findViewById(R.id.recordButton);

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED)
        {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, 1);
        }
        else
        {
            setupMediaRecorder();
        }

        recordButton.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View view)
            {
//                if (mediaRecorder.getState() == MediaRecorder.State.IDLE)
//                {
//                    try
//                    {
//                        mediaRecorder.prepare();
//                        mediaRecorder.start();
//                        recordButton.setText(R.string.stop_recording_button);
//                    }
//                    catch (IOException e)
//                    {
//                        e.printStackTrace();
//                    }
//                }
                if(true)
                {
                    mediaRecorder.stop();
                    mediaRecorder.release();
                    recordButton.setText(R.string.start_recording_button);

                    Toast.makeText(SetupMainActivity.this, "Video saved to: " + outputFile.getAbsolutePath(), Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    private void setupMediaRecorder()
    {
        outputFile = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES), "myVideo.mp4");

        mediaRecorder = new MediaRecorder();
        mediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        mediaRecorder.setVideoSource(MediaRecorder.VideoSource.SURFACE);
        mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        mediaRecorder.setOutputFile(outputFile.getAbsolutePath());
        mediaRecorder.setVideoSize(640, 480);
        mediaRecorder.setVideoFrameRate(30);
        mediaRecorder.setVideoEncodingBitRate(10000000);
        mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
        mediaRecorder.setAudioEncodingBitRate(500000);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults)
    {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == 1 && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED)
        {
            setupMediaRecorder();
        }
        else
        {
            Toast.makeText(this, "Permission denied to record audio.", Toast.LENGTH_SHORT).show();
        }
    }
}
