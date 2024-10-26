# Large V3 Faster Whisper Modal Deployment On Modal.com 

A FastAPI-based server that uses [Faster Whisper](https://github.com/guillaumekln/faster-whisper) for speech-to-text transcription, deployed on [modal.com](https://modal.com). This guide walks you through cloning, setting up, and deploying the server.

---

## Prerequisites

- **Python 3.x**
- **[Modal Account](https://modal.com)** for deployment

---

## Installation Guide

### 1. Clone the Repository

Clone the `faster-whisper-modal` repository to your local machine:

```bash
git clone https://github.com/SYSTRAN/faster-whisper.git
cd faster-whisper-modal
```


### 2. Install the Modal SDK
Install the Modal SDK for deploying applications to the Modal cloud:

```bash
pip install modal
```

### 3. Setup the Modal
Set up Modal authentication. This will open a browser window for you to authorize access to your Modal account:
```bash
python3 -m modal setup
```

### 4. Deploying the App on Modal
Deploy the app on Modal and get the app link from terminal/Modal Dashboard
```bash 
modal deploy app.py    
```

### 5. Test Deployed App:
After the code is deployed, retrieve the app link from the Modal.com Dashboard. The app link will look similar to:

```bash 
curl --location 'https://your-name--faster-whisper-server-fastapi-wrapper.modal.run/transcribe' \
--form 'file=@"/home/user/Desktop/locean-et-lhumanite-destins-lies-lamya-essemlali-tedxorleans-128-ytshorts.savetube.me.mp3"'
```