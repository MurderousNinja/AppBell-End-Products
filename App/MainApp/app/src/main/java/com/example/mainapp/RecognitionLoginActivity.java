package com.example.mainapp;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;

public class RecognitionLoginActivity extends AppCompatActivity {

    private EditText businessIDEditText;
    private EditText businessPasswordEditText;
    private ImageButton togglePasswordButton;
    private boolean isPasswordVisible = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_recognition_login);

        businessIDEditText = findViewById(R.id.businessID_field);
        businessPasswordEditText = findViewById(R.id.password_field);
        togglePasswordButton = findViewById(R.id.toggle_visibility);

        MaterialButton login = findViewById(R.id.login_button);

        togglePasswordButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Toggle password visibility
                isPasswordVisible = !isPasswordVisible;
                togglePasswordVisibility(isPasswordVisible);
            }
        });

        login.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (businessIDEditText.getText().toString().equals("1111") && businessPasswordEditText.getText().toString().equals("11111111")) {
                    Toast.makeText(RecognitionLoginActivity.this, "Login Successful", Toast.LENGTH_SHORT).show();

                    Intent intent = new Intent(RecognitionLoginActivity.this, RecognitionMainActivity.class);
                    startActivity(intent);
                } else {
                    Toast.makeText(RecognitionLoginActivity.this, "UserID, BusinessID, or Business Password is incorrect", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    private void togglePasswordVisibility(boolean isVisible) {
        if (isVisible) {
            businessPasswordEditText.setInputType(1); // InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD
            togglePasswordButton.setImageResource(R.drawable.baseline_visibility_off_24); // Change the icon to hide password
        } else {
            businessPasswordEditText.setInputType(129); // InputType.TYPE_TEXT_VARIATION_PASSWORD
            togglePasswordButton.setImageResource(R.drawable.baseline_visibility_24); // Change the icon to show password
        }

        // Move the cursor to the end of the text to maintain the current position
        businessPasswordEditText.setSelection(businessPasswordEditText.getText().length());
    }
}
