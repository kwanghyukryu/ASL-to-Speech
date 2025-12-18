#include "asl_scraper.h"
#include "hal/joystick.h"
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

// Global flag for clean shutdown
volatile bool g_running = true;

void signal_handler(int signal) {
    if (signal == SIGINT) {
        printf("\n\nShutting down...\n");
        g_running = false;
    }
}

int main(int argc, char* argv[]) {
    // Set up signal handler for Ctrl+C
    signal(SIGINT, signal_handler);
    
    // Default URL - Mac USB network IP
    const char* api_url = "http://192.168.7.1:5002";
    int interval = 100;  // milliseconds
    
    // Parse command-line arguments
    if (argc > 1) {
        api_url = argv[1];
    }
    if (argc > 2) {
        interval = atoi(argv[2]);
        if (interval <= 0) {
            interval = 100;
        }
    }
    
    // Initialize joystick
    printf("Initializing joystick...\n");
    Joystick_init();
    
    // Create scraper
    ASLDataScraper* scraper = asl_scraper_create(api_url);
    if (!scraper) {
        fprintf(stderr, "Failed to create scraper\n");
        Joystick_cleanup();
        return 1;
    }
    
    printf("\n================================\n");
    printf("ASL Data Scraper with Joystick\n");
    printf("================================\n");
    printf("Fetching from: %s\n", api_url);
    printf("Update interval: %dms\n", interval);
    printf("\nJoystick Controls:\n");
    printf("  UP    - Save current letter\n");

    printf("  DOWN  - Add space\n");
    printf("  LEFT  - Speak current line\n");
    printf("  RIGHT - Add newline\n");
    printf("  Ctrl+C - Exit\n\n");
    
    sleep(2);
    
    // Track previous joystick state to detect new presses
    JoystickDirection prev_direction = JOYSTICK_NONE;
    
    // Main loop
    while (g_running) {
        // Fetch and display current letter
        char* letter = asl_scraper_fetch_letter(scraper);
        asl_scraper_display_letter(scraper, letter);
        
        // Read joystick
        JoystickDirection current_direction = Joystick_get_direction();
        
        // Detect new press (edge detection - only trigger on direction change)
        if (current_direction != JOYSTICK_NONE && current_direction != prev_direction) {
            switch (current_direction) {
                case JOYSTICK_UP:
                    asl_scraper_save_letter(scraper);
                    usleep(300000);  // Debounce delay
                    break;
                    
                case JOYSTICK_DOWN:
                    asl_scraper_add_space();
                    usleep(500000);  // Debounce delay
                    break;
                    
                case JOYSTICK_RIGHT:
                    asl_scraper_add_newline();
                    usleep(500000);  // Debounce delay
                    break;
                    
                case JOYSTICK_LEFT:
                    speak_last_prediction();
                    usleep(500000);  // Debounce delay
                    break;
                    
                default:
                    break;
            }
        }
        
        prev_direction = current_direction;
        
        // Wait before next iteration
        usleep(interval * 1000);  // Convert ms to microseconds
    }
    
    // Cleanup
    asl_scraper_destroy(scraper);
    Joystick_cleanup();
    printf("Cleanup complete.\n");
    printf("Goodbye!\n");
    
    return 0;
}

/*
COMPILATION:
============
On BeagleY-AI:

gcc -std=c11 main.c asl_scraper.c \
    /home/username/cmake_starter/hal/src/joystick.c \
    -I/home/username/cmake_starter/hal/include \
    -lcurl -o asl_scraper_joystick

USAGE:
======
  ./asl_scraper_joystick                              # Default: 192.168.7.1:5002
  ./asl_scraper_joystick http://192.168.7.1:5002      # Custom URL
  ./asl_scraper_joystick http://192.168.7.1:5002 50   # Custom URL + 50ms interval

JOYSTICK CONTROLS:
==================
  UP    - Save current ASL letter to asl_predictions.txt
  DOWN  - Add space to file
  RIGHT - Add newline to file
  LEFT  - (Reserved for future use)
  
  Ctrl+C - Exit program

NOTES:
======
- Make sure f3.py is running on your Mac first
- Verify network connectivity: ping 192.168.7.1
- Test endpoint: curl http://192.168.7.1:5002
- Output file: asl_predictions.txt (created in current directory)
- Make sure to update the joystick.c path to your actual location

EXAMPLE WORKFLOW:
=================
1. On Mac:
   python3 f3.py
   
2. On BeagleY (via SSH):
   ./asl_scraper_joystick
   
3. Make ASL signs in front of Mac webcam
4. Use joystick on BeagleY to save letters
5. Check asl_predictions.txt for your captured text
*/

