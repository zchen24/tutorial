//
// Sample LED
//

#ifndef ECAT_DEFS_LED_H
#define ECAT_DEFS_LED_H

#define k_LEDVendorId 0x34E
#define k_LEDProductId 0xAA

typedef struct LEDOutput
{
    u_char LED1:1;  // 1-bit LED1
    u_char LED2:1;
    u_char LED3:1;
    u_char LED4:1;
    u_char LED5:1;
    u_char LED6:1;
    u_char LED7:1;
    u_char LED8:1;
} LEDOutput_t;

typedef struct LEDInput
{
    char BTN1:1;
    char BTN2:1;
} LEDInput_t;

#endif //ECAT_DEFS_LED_H
