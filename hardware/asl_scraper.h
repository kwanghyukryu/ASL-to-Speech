#ifndef ASL_SCRAPER_H
#define ASL_SCRAPER_H

#include <stdbool.h>
#include <stddef.h>

// Opaque pointer to scraper structure
typedef struct ASLDataScraper ASLDataScraper;

// Callback function for CURL to write data
size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp);

// Create and destroy scraper
ASLDataScraper* asl_scraper_create(const char* api_url);
void asl_scraper_destroy(ASLDataScraper* scraper);

// Core functions
char* asl_scraper_fetch_letter(ASLDataScraper* scraper);
void asl_scraper_display_letter(ASLDataScraper* scraper, const char* letter);
void asl_scraper_save_letter(ASLDataScraper* scraper);
void asl_scraper_add_space(void);
void asl_scraper_add_newline(void);
void print_last_prediction(void);
void speak_last_prediction(void);

// Getters
const char* asl_scraper_get_last_letter(ASLDataScraper* scraper);

#endif // ASL_SCRAPER_H
