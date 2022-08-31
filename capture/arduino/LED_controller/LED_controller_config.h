/*  Configuration data for 240-channel LED controller.
 *  Written by Mark Stambaugh (mstambaugh@ucsd.edu) for Salk Azim Lab.
 *  Last edited 2021/08/05
 */

#ifndef LED_CONTROLLER_CONFIG_H
#define LED_CONTROLLER_CONFIG_H

//TODO update all of this

#define UV_MA_MAX 4000
#define UV_MA_MIN 0
#define VIS_MA_MAX 4000
#define VIS_MA_MIN 0
#define FRAME_LENGTH_US_MAX 50000
#define FRAME_LENGTH_US_MIN 10
#define FRAME_COUNT_MAX 100000
#define FRAME_COUNT_MIN 1

#define VS_UV_STARTUP_TIMEOUT_MS 5000
#define VS_UV_IDLE_CURRENT_MA    50

#define VS_UV_SHUTDOWN_TIMEOUT_MS 5000
#define VS_UV_SHUTDOWN_MV         2000

#define VS_VIS_STARTUP_TIMEOUT_MS 5000
#define VS_VIS_IDLE_CURRENT_MA    50

#define VS_VIS_SHUTDOWN_TIMEOUT_MS 5000
#define VS_VIS_SHUTDOWN_MV         2000

#define CHECK_LED_UV_CARD_COUNT 10
#define CHECK_LED_VIS_CARD_COUNT 10
#define CHECK_LED_OVERSAMPLE_COUNT 100
#define VIS_LED_CHECK_CURRENT_MA 50
#define UV_LED_CHECK_CURRENT_MA 50


#define VIS_LED_TEST_CURRENT_MA 10
#define UV_LED_TEST_CURRENT_MA 10



#define TEST_DELAY_MS 200

typedef struct Parameters{
  uint16_t UV_mA;
  uint16_t VIS_mA;
  byte UV_en[20];
  byte VIS_en[20];
  byte camera_en;
  uint32_t camera_VIS_on_us;
  uint32_t LED_VIS_off_us;
  uint32_t LED_UV_on_us;
  uint32_t LED_UV_off_us;
  uint32_t camera_UV_on_us;
  uint32_t frame_length_us;
  uint32_t frame_count;
  
};

Parameters params_default = {10,  //UV_mA 
                             10,  //VIS_mA
                             0x02,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, //UV_en
                             0x02,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, //VIS_en
                             0xFF, //camera_en
                             2500, //camera_VIS_on_us
                             5000, //LED_VIS_off_us
                             5050, //LED_UV_on_us
                             7500, //LED_UV_off_us
                             7550, //camera_UV_on_us
                             10000, //frame_length_us
                             100 //frame_count
                             };

Parameters params_test = {10,  //UV_mA //TODO initialize new fields
                          10,  //VIS_mA
                          0x02,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, //UV_en
                          0x02,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, //VIS_en
                          0, //camera_en, not used in test procedure
                          0, //camera_VIS_on_us, not used in test procedure
                          0, //LED_VIS_off_us, not used in test procedure
                          0, //LED_UV_on_us, not used in test procedure
                          0, //LED_UV_off_us, not used in test procedure
                          0, //camera_UV_on_us, not used in test procedure
                          0, //frame_length_us, not used in test procedure
                          0 //frame_count, not used in test procedure
                          }; 



#endif //LED_CONTROLLER_CONFIG_H
