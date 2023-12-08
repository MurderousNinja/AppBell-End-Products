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

## Description

### MainActivity

The MainActivity serves as the entry point of the application. It contains two buttons, leftButton and rightButton, each directing users to different activities.

### SetupSignupActivity

The `SetupSignupActivity` is a crucial component of the Android application for business setup. It allows users to input their business ID, password, and includes a toggle button for password visibility. The main feature is the Login button, triggering navigation to the `SetupLoginActivity` upon click, streamlining the user authentication process.

### SetupLoginActivity

The `SetupLoginActivity` class in the "mainapp" package of your Android application defines the user authentication screen. This activity presents fields for entering a business ID and password, with an optional password visibility toggle. Upon successful login with predefined credentials, users are redirected to the `SetupMainActivity`. The login logic is implemented in the click listener for the login button, while the password visibility toggle is managed by the `togglePasswordVisibility` method. The activity enhances user experience by allowing password visibility customization and provides feedback through toast messages for login success or failure.

### SetupMainActivity

The `SetupMainActivity` class in the "mainapp" package of your Android application serves as the primary interface for video recording and camera control. Users can capture videos with features such as flash toggle, camera flip, and a countdown timer displayed on the UI. The activity employs the CameraX library for camera functionality and provides a seamless recording experience with audio capabilities. Recorded videos are saved with predefined settings in the "Documents/MainApp" directory. Additionally, the activity integrates a timer to automatically stop recording after a set duration, enhancing usability for users capturing short videos.


### RecognitionLoginActivity

The `RecognitionLoginActivity` class, residing in the "mainapp" package of your Android application, defines the user authentication screen for recognition-based login. Users input a business ID and password, with an option to toggle password visibility. Upon successful login using predefined credentials, the activity navigates users to the `RecognitionMainActivity`. The login logic is implemented in the click listener for the login button, and the password visibility toggle is managed through the `togglePasswordVisibility` method, enhancing user experience by providing visual feedback for password entry.

### RecognitionMainActivity

The `RecognitionMainActivity` class in the "mainapp" package of your Android application serves as the core functionality for video recognition processing. Users can select a video file and initiate the recognition process by providing a business ID and clicking the start button. The activity then asynchronously performs an API call to a specified endpoint, sending the selected video along with the business ID as parameters. The API response is displayed in the UI, providing feedback on the recognition process. The code leverages an AsyncTask to handle the API call in the background, ensuring a smooth user experience during video processing.
