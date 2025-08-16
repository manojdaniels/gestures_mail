# Hand Gesture Email Reader App

This application allows you to read Gmail emails using hand gestures captured through your webcam.

## Features

- **Hand Gesture Recognition**: Uses MediaPipe for real-time hand tracking
- **Gmail Integration**: Fetches unread emails from your Gmail account
- **Gesture Controls**:
  - **Pinch (thumb + index finger)**: Zoom in/out
  - **Pinch (thumb + middle finger)**: Drag the email panel
  - **Swipe left/right**: Navigate between emails

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Gmail API Credentials

To use this app, you need to set up Gmail API credentials:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Gmail API for your project
4. Go to "Credentials" → "Create Credentials" → "OAuth 2.0 Client IDs"
5. Choose "Desktop application" as the application type
6. Download the credentials JSON file
7. Rename it to `credentials.json` and place it in the same directory as `gesture_mail.py`

**Important**: The `credentials.json` file should look like this (with your actual values):

```json
{
  "installed": {
    "client_id": "YOUR_ACTUAL_CLIENT_ID.apps.googleusercontent.com",
    "project_id": "YOUR_ACTUAL_PROJECT_ID",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_secret": "YOUR_ACTUAL_CLIENT_SECRET",
    "redirect_uris": ["http://localhost"]
  }
}
```

### 3. First Run Authentication

On the first run, the application will:
1. Open your default browser to authorize the application
2. Ask you to sign in to your Google account
3. Request permission to read your Gmail
4. Create a `token.json` file for future use

### 4. Run the Application

```bash
python gesture_mail.py
```

## Usage

- **Quit**: Press 'q' key
- **Navigate emails**: Swipe left/right with your hand
- **Zoom**: Pinch with thumb and index finger
- **Move panel**: Pinch with thumb and middle finger

## Troubleshooting

- **"Could not open webcam"**: Make sure your webcam is connected and not being used by another application
- **Authentication errors**: Delete `token.json` and ensure `credentials.json` is properly set up
- **Gmail API errors**: Check that the Gmail API is enabled in your Google Cloud project

## Requirements

- Python 3.7+
- Webcam
- Gmail account
- Google Cloud project with Gmail API enabled

"# gestures_mail" 
