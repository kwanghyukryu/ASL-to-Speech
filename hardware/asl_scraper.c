#include "asl_scraper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <unistd.h>
#include "hal/joystick.h"
#define MAX_RESPONSE_SIZE 1024
#define MAX_LETTER_SIZE 256

// Internal structure definition
struct ASLDataScraper {
    char* url;
    CURL* curl;
    char last_letter[MAX_LETTER_SIZE];
};

// Structure for CURL write callback
typedef struct {
    char* data;
    size_t size;
} ResponseBuffer;

// Callback function for CURL to write data
size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total_size = size * nmemb;
    ResponseBuffer* buffer = (ResponseBuffer*)userp;
    
    size_t new_size = buffer->size + total_size;
    if (new_size >= MAX_RESPONSE_SIZE - 1) {
        return 0; // Buffer overflow protection
    }
    
    memcpy(buffer->data + buffer->size, contents, total_size);
    buffer->size = new_size;
    buffer->data[buffer->size] = '\0';
    
    return total_size;
}

ASLDataScraper* asl_scraper_create(const char* api_url) {
    ASLDataScraper* scraper = (ASLDataScraper*)malloc(sizeof(ASLDataScraper));
    if (!scraper) {
        return NULL;
    }
    
    // Allocate and copy URL
    scraper->url = (char*)malloc(strlen(api_url) + 1);
    if (!scraper->url) {
        free(scraper);
        return NULL;
    }
    strcpy(scraper->url, api_url);
    
    // Initialize CURL
    curl_global_init(CURL_GLOBAL_DEFAULT);
    scraper->curl = curl_easy_init();
    
    if (!scraper->curl) {
        free(scraper->url);
        free(scraper);
        return NULL;
    }
    
    // Initialize last letter
    strcpy(scraper->last_letter, "None");
    
    return scraper;
}

void asl_scraper_destroy(ASLDataScraper* scraper) {
    if (scraper) {
        if (scraper->curl) {
            curl_easy_cleanup(scraper->curl);
        }
        if (scraper->url) {
            free(scraper->url);
        }
        free(scraper);
    }
    curl_global_cleanup();
}

char* asl_scraper_fetch_letter(ASLDataScraper* scraper) {
    if (!scraper || !scraper->curl) {
        strcpy(scraper->last_letter, "ERROR");
        return scraper->last_letter;
    }
    
    // Allocate response buffer
    char* response_data = (char*)malloc(MAX_RESPONSE_SIZE);
    if (!response_data) {
        strcpy(scraper->last_letter, "ERROR");
        return scraper->last_letter;
    }
    
    ResponseBuffer buffer = {
        .data = response_data,
        .size = 0
    };
    response_data[0] = '\0';
    
    // Set CURL options
    curl_easy_setopt(scraper->curl, CURLOPT_URL, scraper->url);
    curl_easy_setopt(scraper->curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(scraper->curl, CURLOPT_WRITEDATA, &buffer);
    curl_easy_setopt(scraper->curl, CURLOPT_TIMEOUT, 1L);
    
    // Perform request
    CURLcode res = curl_easy_perform(scraper->curl);
    
    if (res != CURLE_OK) {
        strcpy(scraper->last_letter, "ERROR");
        free(response_data);
        return scraper->last_letter;
    }
    
    // Copy response to last_letter
    strncpy(scraper->last_letter, response_data, MAX_LETTER_SIZE - 1);
    scraper->last_letter[MAX_LETTER_SIZE - 1] = '\0';
    
    free(response_data);
    return scraper->last_letter;
}

void asl_scraper_display_letter(ASLDataScraper* scraper, const char* letter) {
    // Clear screen
    (void)scraper; // Unused parameter
    printf("\033[2J\033[1;1H");   
    printf("\n\n");
    printf("  Current Letter: ");
    
    if (strcmp(letter, "None") == 0 || strcmp(letter, "ERROR") == 0) {
        printf("\033[1;33m%s\033[0m", letter);  // Yellow
    } else {
        printf("\033[1;32m%s\033[0m", letter);  // Green
    }
    
    printf("\n\n");
    printf("  Joystick Controls:\n");
    printf("  UP - Save letter to file\n");
    printf("  DOWN - Add space\n");
    printf("  RIGHT - Add newline\n");
    printf("  LEFT  - Speak current line\n");
    printf("  Ctrl+C to exit\n\n");
    fflush(stdout);
}

void speak_last_prediction() {
    const char *path = "/tmp/asl_predictions.txt";

    FILE* file = fopen(path, "r");
    if (!file) {
        printf("\n ✘ Could not open prediction file (%s)\n", path);
        return;
    }

    char line[256];
    char last_line[256] = "";

    // Read every line, keep only the last one
    while (fgets(line, sizeof(line), file)) {
        strcpy(last_line, line);
    }

    fclose(file);

    // Trim newline
    last_line[strcspn(last_line, "\n")] = 0;

    if (strlen(last_line) == 0) {
        printf("\n ⚠ Prediction file exists, but is empty.\n");
        return;
    }

    printf("\n ✓ Speaking: %s\n", last_line);
    fflush(stdout);

    // ---- Speak using eSpeak ----
    char command[300];
    snprintf(command, sizeof(command),
             "espeak \"%s\" 2>/dev/null", last_line);

    system(command);
}

void print_last_prediction() {
    const char* path = "/tmp/asl_predictions.txt";
    FILE* file = fopen(path, "r");

    if (!file) {
        printf("\n ✘ Could not open %s\n", path);
        return;
    }

    char line[256];
    char last_line[256] = "";

    // Read the file line by line
    while (fgets(line, sizeof(line), file)) {
        strcpy(last_line, line);
    }

    fclose(file);

    // Remove newline if present
    last_line[strcspn(last_line, "\n")] = 0;

    if (strlen(last_line) == 0) {
        printf("\n ⚠ File exists, but no predictions saved yet.\n");
    } else {
        printf("\n ✓ Last Prediction: %s\n", last_line);
    }
}


void asl_scraper_save_letter(ASLDataScraper* scraper) {
    if (!scraper) {
        return;
    }
    
    if (strcmp(scraper->last_letter, "None") == 0 || 
        strcmp(scraper->last_letter, "ERROR") == 0) {
        printf("\n  ✗ No letter to save\n");
        fflush(stdout);
        usleep(300000);  // 300ms
        return;
    }

    FILE* file = fopen("/tmp/asl_predictions.txt", "a");
    if (file) {
        fprintf(file, "%s", scraper->last_letter);
        fclose(file);
        printf("\n  ✓ Saved: %s\n", scraper->last_letter);
    } else {
        printf("\n  ✗ Error opening file\n");
    }
    fflush(stdout);
    usleep(300000);  // 300ms
}

void asl_scraper_add_space(void) {
    FILE* file = fopen("/tmp/asl_predictions.txt", "a");
    if (file) {
        fprintf(file, " ");
        fclose(file);
        printf("\n  ✓ Added space\n");
    } else {
        printf("\n  ✗ Error opening file\n");
    }
    fflush(stdout);
    usleep(300000);  // 300ms
}

void asl_scraper_add_newline(void) {
    FILE* file = fopen("/tmp/asl_predictions.txt", "a");
    if (file) {
        fprintf(file, "\n");
        fclose(file);
        printf("\n  ✓ Added newline\n");
    } else {
        printf("\n  ✗ Error opening file\n");
    }
    fflush(stdout);
    usleep(300000);  // 300ms
}

const char* asl_scraper_get_last_letter(ASLDataScraper* scraper) {
    return scraper ? scraper->last_letter : "ERROR";
}
