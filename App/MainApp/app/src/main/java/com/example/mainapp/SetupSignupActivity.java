package com.example.mainapp;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;

public class SetupSignupActivity extends AppCompatActivity {

    private EditText businessIDEditText;
    private EditText businessPasswordEditText;
    private ImageButton togglePasswordButton;
    private Button LoginButton;
    private Button SignupButton;
    private boolean isPasswordVisible = false;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_setup_signup);

        businessIDEditText = findViewById(R.id.businessID_field);
        LoginButton = findViewById(R.id.login_button);

        LoginButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(SetupSignupActivity.this, SetupLoginActivity.class);
                startActivity(intent);
            }
        });
    }
}