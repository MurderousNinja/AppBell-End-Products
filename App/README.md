# Readme for Main App

## Features

- Signup Page
- Login Page
- Camera Page
- Video Selector Page

## Screenshots

<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/MurderousNinja/AppBell-End-Products/blob/main/App/Screenshots/Home-Page.jpg" alt="Description of your image 1" width="200" height="433.898">
  <img src="https://github.com/MurderousNinja/AppBell-End-Products/blob/main/App/Screenshots/SignIn-Page.jpg" alt="Description of your image 5" width="200" height="433.808">
  <img src="https://github.com/MurderousNinja/AppBell-End-Products/blob/main/App/Screenshots/SingUp-Page.jpg" alt="Description of your image 6" width="200" height="435.587">
</div>

<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/MurderousNinja/AppBell-End-Products/blob/main/App/Screenshots/Setup-Main.jpg" alt="Description of your image 3" width="200" height="435.587">
  <img src="https://github.com/MurderousNinja/AppBell-End-Products/blob/main/App/Screenshots/Setup-Processing.jpg" alt="Description of your image 4" width="200" height="434.876">
  <img src="https://github.com/MurderousNinja/AppBell-End-Products/blob/main/App/Screenshots/Recognition-Main.jpg" alt="Description of your image 2" width="200" height="434.876">
</div>

## Getting Started

Providing instructions on how to set up and running the app locally.

### Prerequisites

  - Android Studio Giraffe | 2022.3.1
  - Gradle 8.2

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/MurderousNinja/AppBell-Main-App.git
   ```

2. Open the project in Android Studio.

3. Build and run the app on an emulator or a physical device.

# Description

## MainActivity
- **Role**: Entry point of the application.
- **Components**: 
  - Two buttons: `leftButton` and `rightButton`.
  - Directs users to different activities.

## SetupSignupActivity
- **Role**: Crucial for business setup in the Android application.
- **Features**:
  - Input fields for business ID and password.
  - Toggle button for password visibility.
  - Login button for navigation to `SetupLoginActivity`.
  
## SetupLoginActivity
- **Role**: User authentication screen.
- **Components**:
  - Fields for business ID and password.
  - Password visibility toggle.
  - Login button triggering navigation to `SetupMainActivity`.
- **Logic**:
  - Implements login logic in the click listener for the login button.
  - Manages password visibility through `togglePasswordVisibility` method.
  - Toast messages for login success or failure.

## SetupMainActivity
- **Role**: Primary interface for video recording and camera control.
- **Features**:
  - Camera functionality using CameraX library.
  - Video capture with flash toggle, camera flip, and countdown timer.
  - Saves recorded videos in "Documents/MainApp" directory.
  - Automatic stop recording with a timer.
  
## RecognitionLoginActivity
- **Role**: User authentication screen for recognition-based login.
- **Components**:
  - Fields for business ID and password.
  - Password visibility toggle.
  - Login button for navigation to `RecognitionMainActivity`.
- **Logic**:
  - Implements login logic in the click listener for the login button.
  - Manages password visibility through `togglePasswordVisibility` method.
  - Visual feedback for password entry.

## RecognitionMainActivity
- **Role**: Core functionality for video recognition processing.
- **Features**:
  - Selects a video file.
  - Initiates recognition with business ID and start button.
  - Asynchronously performs API call to a specified endpoint.
  - Displays API response in the UI for recognition feedback.
  - Uses AsyncTask for background API call, ensuring a smooth user experience during video processing.

# Notes

- While developing the Android app, it's important to note that I'm not an experienced Android developer. Consequently, there might be existing bugs, inefficiencies, or suboptimal logic within the application.

- Unfortunately, I faced challenges integrating Python into the Android Studio app. Despite my attempts, I couldn't successfully incorporate Python functionality as initially planned.

- As an alternative, I considered using APIs to replace the internal Python compilation. This shift aims to enhance the app's performance and leverage external services for improved functionality.
