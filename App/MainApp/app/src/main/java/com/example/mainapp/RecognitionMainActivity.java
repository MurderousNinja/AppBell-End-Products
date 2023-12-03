package com.example.mainapp;

import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class RecognitionMainActivity extends AppCompatActivity {
    private Button btnFilePicker;
    private TextView textView1;
    private Button startBtn;
    private static final int FILE_PICKER_REQUEST_CODE = 100;
    private Uri videoUri;
    private EditText businessID;
    private String url = "https://604f-103-200-106-17.ngrok-free.app" + "/process_video"; // Replace with your actual API endpoint URL

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_recognition_main);

        btnFilePicker = findViewById(R.id.file_picker_button);
        textView1 = findViewById(R.id.text_view1);
        startBtn = findViewById(R.id.start_button);
        businessID = findViewById(R.id.businessID_field);

        btnFilePicker.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                showFileChooser();
            }
        });

        startBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Call the API
                callApi();
            }
        });
    }

    private void showFileChooser() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("video/*"); // Specify the file type to video files
        intent.addCategory(Intent.CATEGORY_OPENABLE);

        try {
            startActivityForResult(Intent.createChooser(intent, "Select a video file"), FILE_PICKER_REQUEST_CODE);
        } catch (Exception exception) {
            Toast.makeText(RecognitionMainActivity.this, "Please Install a File Manager", Toast.LENGTH_SHORT).show();
        }
    }

    private void callApi() {
        // Check if a video has been selected
        if (videoUri == null) {
            Toast.makeText(RecognitionMainActivity.this, "Please select a video first", Toast.LENGTH_SHORT).show();
            return;
        }

        // Get the text from the businessID EditText
        final String bid = businessID.getText().toString();

        // Use an AsyncTask to perform the API call in the background
        new MyAsyncTask(url, bid, videoUri).execute();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == FILE_PICKER_REQUEST_CODE && resultCode == RESULT_OK && data != null) {
            videoUri = data.getData();
            if (videoUri != null) {
                // Get the selected video's name from the URI
                String videoName = getVideoNameFromUri(videoUri);

                // Set the video name in the TextView
                textView1.setText("Selected Video: " + videoName);
            }
        }
    }

    private class MyAsyncTask extends AsyncTask<Void, Void, String> {
        private final String url;
        private final String bid;
        private final Uri videoUri;

        MyAsyncTask(String url, String bid, Uri videoUri) {
            this.url = url;
            this.bid = bid;
            this.videoUri = videoUri;
        }

        @Override
        protected String doInBackground(Void... voids) {
            try {
                // Create a connection to the API endpoint
                URL apiUrl = new URL(url);
                HttpURLConnection connection = (HttpURLConnection) apiUrl.openConnection();
                connection.setRequestMethod("POST");
                connection.setDoOutput(true);

                // Create the output stream for the request
                DataOutputStream outputStream = new DataOutputStream(connection.getOutputStream());

                // Add the business ID as a parameter
                outputStream.writeBytes("bid=" + bid + "&");

                // Add the video file as a parameter
                String boundary = "*****";
                String lineEnd = "\r\n";
                String twoHyphens = "--";
                String attachmentName = "video_file";
                String crlf = "\r\n";

                outputStream.writeBytes("Content-Disposition: form-data; name=\"" + attachmentName + "\"; filename=\"" + getVideoNameFromUri(videoUri) + "\"" + crlf);
                outputStream.writeBytes("Content-Type: " + "video/*" + crlf);
                outputStream.writeBytes(crlf);

                // Read and write the video file data
                InputStream inputStream = RecognitionMainActivity.this.getContentResolver().openInputStream(videoUri);
                byte[] buffer = new byte[1024];
                int bytesRead;
                while (true) {
                    assert inputStream != null;
                    if ((bytesRead = inputStream.read(buffer)) == -1) break;
                    outputStream.write(buffer, 0, bytesRead);
                }
                inputStream.close();

                // Write the closing boundary
                outputStream.writeBytes(crlf);
                outputStream.writeBytes(twoHyphens + boundary + twoHyphens + crlf);

                // Flush and close the output stream
                outputStream.flush();
                outputStream.close();

                // Get the response from the API
                int responseCode = connection.getResponseCode();
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    // Read and handle the API response
                    InputStream responseStream = connection.getInputStream();
                    return readResponseString(responseStream);
                } else {
                    // Handle the error
                    return "Error: HTTP " + responseCode;
                }
            } catch (IOException e) {
                e.printStackTrace();
                return "Error: " + e.getMessage();
            }
        }

        @Override
        protected void onPostExecute(String result) {
            super.onPostExecute(result);
            // Update the TextView with the API response
            textView1.setText("API Response: " + result);
        }

        // Helper method to read the response from an InputStream
        private String readResponseString(InputStream inputStream) throws IOException {
            StringBuilder response = new StringBuilder();
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            String line;
            while ((line = reader.readLine()) != null) {
                response.append(line);
            }
            reader.close();
            return response.toString();
        }
    }

    // Helper method to get the video file name from a URI
    private String getVideoNameFromUri(Uri videoUri) {
        String videoName = null;
        String[] projection = {MediaStore.Video.Media.DISPLAY_NAME};
        Cursor cursor = getContentResolver().query(videoUri, projection, null, null, null);

        if (cursor != null && cursor.moveToFirst()) {
            int columnIndex = cursor.getColumnIndexOrThrow(MediaStore.Video.Media.DISPLAY_NAME);
            videoName = cursor.getString(columnIndex);
            cursor.close();
        }

        return videoName;
    }
}
