# ASL Hand Sign to Text and Speech System

## Overview

This project implements a real-time American Sign Language (ASL) hand sign recognition system that converts detected ASL letters into text and speech. The system uses a distributed architecture in which a Python-based ASL recognition server runs on a Mac, while a BeagleY-AI board acts as a client that retrieves predictions over HTTP and provides physical interaction through a joystick.

The project integrates computer vision, network communication, embedded C programming, and hardware input to create an accessible and interactive ASL interface.

---

## System Architecture

The system operates as a client–server pipeline:

1. A webcam connected to a Mac captures hand images.
2. A Python server processes video frames and performs ASL letter recognition.
3. The BeagleY-AI board periodically fetches predictions using HTTP requests.
4. A joystick connected to the BeagleY-AI allows the user to save letters, insert spaces or newlines, and trigger text-to-speech output.
5. Saved text is written to a persistent file and can be spoken aloud using eSpeak.

---

## Features

* Real-time ASL letter retrieval over HTTP
* Hardware-based user input via joystick
* Persistent text output stored in a file
* Text-to-speech playback using eSpeak
* Configurable polling interval
* Clean shutdown handling using SIGINT

---

## Joystick Controls

| Direction | Action                  |
| --------- | ----------------------- |
| UP        | Save current ASL letter |
| DOWN      | Add a space             |
| RIGHT     | Add a newline           |
| LEFT      | Speak the current line  |
| Ctrl+C    | Exit the program        |

---

## Project Structure

```
.
├── main.c            # Main application loop and joystick handling
├── asl_scraper.c     # HTTP fetching, display logic, file I/O, and speech
├── asl_scraper.h     # ASL scraper interface definitions
├── hal/
│   ├── joystick.c    # SPI-based joystick driver implementation
│   └── joystick.h    # Joystick interface definitions
├── f3.py             # Python ASL recognition server (runs on Mac)
└── README.md
```

---

## Dependencies

### BeagleY-AI Requirements

* GCC with C11 support
* libcurl (for HTTP communication)
* eSpeak (for text-to-speech output)
* SPI enabled for joystick input

Install required packages:

```bash
sudo apt update
sudo apt install libcurl4-openssl-dev espeak
```

---

## Compilation

Compile the program directly on the BeagleY-AI board:

```bash
gcc -std=c11 main.c asl_scraper.c \
    /home/username/cmake_starter/hal/src/joystick.c \
    -I/home/username/cmake_starter/hal/include \
    -lcurl -o asl_scraper_joystick
```

Ensure that the joystick source and include paths match your local directory structure.

---

## Usage

### Step 1: Start the ASL Server on the Mac

```bash
python3 f3.py
```

### Step 2: Run the Client on the BeagleY-AI

```bash
./asl_scraper_joystick
```

Optional arguments:

```bash
./asl_scraper_joystick <API_URL> <INTERVAL_MS>
```

Example:

```bash
./asl_scraper_joystick http://192.168.7.1:5002 50
```

---

## Output

All captured text is written to the following file:

```
/tmp/asl_predictions.txt
```

Each line in the file represents a sentence that can be replayed using the text-to-speech function.

---

## Debugging and Testing

* Verify network connectivity:

```bash
ping 192.168.7.1
curl http://192.168.7.1:5002
```

* Ensure SPI is enabled on the BeagleY-AI
* Confirm joystick wiring and SPI channel configuration
* Ensure the Python server is running before launching the client

---

## Future Improvements

* Support for word-level or sentence-level ASL recognition
* Graphical user interface instead of terminal output
* Offline text-to-speech support
* On-device ASL inference without network dependency

---

## Author

This project was developed as an embedded systems and computer vision application demonstrating integration of C programming, hardware interfaces, networking, and assistive technology design.
