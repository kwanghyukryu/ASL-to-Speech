#include "hal/joystick.h"
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <linux/spi/spidev.h>
#include <stdint.h>

#define SPI_DEVICE "/dev/spidev0.0"
#define SPI_MODE 0
#define SPI_BITS_PER_WORD 8
#define SPI_SPEED_HZ 250000

#define JOYSTICK_X_CHANNEL 0
#define JOYSTICK_Y_CHANNEL 1

#define MIN_X 1000
#define MAX_X 3600
#define CENTER_X 2000
#define MIN_Y 1000
#define MAX_Y 3600
#define CENTER_Y 2050

#define DIRECTION_THRESHOLD 0.70

static int spi_fd = -1;

static int min_x = MIN_X;
static int max_x = MAX_X;
static int min_y = MIN_Y;
static int max_y = MAX_Y;
static int center_x = CENTER_X;
static int center_y = CENTER_Y;


static int read_ch(int fd, int ch, uint32_t speed_hz)
{
    uint8_t tx[3] = {
        (uint8_t)(0x06 | ((ch & 0x04) >> 2)),
        (uint8_t)((ch & 0x03) << 6),
        0x00
    };
    uint8_t rx[3] = { 0 };

    struct spi_ioc_transfer tr = {
        .tx_buf = (unsigned long)tx,
        .rx_buf = (unsigned long)rx,
        .len = 3,
        .speed_hz = speed_hz,
        .bits_per_word = 8,
        .cs_change = 0
    };

    if (ioctl(fd, SPI_IOC_MESSAGE(1), &tr) < 1) {
        perror("SPI read failed");
        return -1;
    }

    return ((rx[1] & 0x0F) << 8) | rx[2];  
}


void Joystick_init(void)
{

    spi_fd = open(SPI_DEVICE, O_RDWR);

}

JoystickDirection Joystick_get_direction(void)
{
    int x = read_ch(spi_fd, JOYSTICK_X_CHANNEL,SPI_SPEED_HZ);
    int y = read_ch(spi_fd, JOYSTICK_Y_CHANNEL,SPI_SPEED_HZ);

    if (x < 0 || y < 0) {
        return JOYSTICK_NONE;
    }

    int range_x = (max_x - min_x) / 2;
    int range_y = (max_y - min_y) / 2;

    if (range_x == 0) range_x = 1;
    if (range_y == 0) range_y = 1;

    int offset_x = x - center_x;
    int offset_y = y - center_y;

    int threshold_x = (int)(range_x * DIRECTION_THRESHOLD);
    int threshold_y = (int)(range_y * DIRECTION_THRESHOLD);

    int abs_offset_x = (offset_x < 0) ? -offset_x : offset_x;
    int abs_offset_y = (offset_y < 0) ? -offset_y : offset_y;

    if (abs_offset_x > abs_offset_y) {
        if (abs_offset_x > threshold_x) {
            return (offset_x > 0) ? JOYSTICK_RIGHT : JOYSTICK_LEFT;
        }
    } else {
        if (abs_offset_y > threshold_y) {
            return (offset_y > 0) ? JOYSTICK_UP : JOYSTICK_DOWN;
        }
    }

    return JOYSTICK_NONE;
}

bool Joystick_is_pressed(void)
{
    return Joystick_get_direction() != JOYSTICK_NONE;
}

void Joystick_cleanup(void)
{
    close(spi_fd);
}