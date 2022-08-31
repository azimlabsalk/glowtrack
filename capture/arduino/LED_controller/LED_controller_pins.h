/*  Pin definitions for the 240-channel LED controller.
 *  Written by Mark Stambaugh (mstambaugh@ucsd.edu) for Salk Azim Lab.
 *  Last edited 2021/05/19
 */

#ifndef LED_CONTROLLER_PINS_H
#define LED_CONTROLLER_PINS_H

//data shift register controls
#define PIN_SH_CLK_UV     2 //PD2
#define PIN_SH_DIN_UV     3 //PD3

#define PIN_SH_CLK_VIS    8 //PB0
#define PIN_SH_DIN_VIS    4 //PD4

#define PIN_LED_LATCH     5 //PD5
#define PIN_OUT_RSTn      6 //PD6
#define PIN_SH_RSTn       7 //PD7

#define PIN_CAMERA_LATCH  13 //PB5

//direct port assignments for faster setting
#define PORT_SH_CLK_UV    PORTD
#define PORT_SH_DIN_UV    PORTD
#define PORT_SH_CLK_VIS   PORTB
#define PORT_SH_DIN_VIS   PORTD
#define PORT_LED_LATCH    PORTD
#define PORT_OUT_RSTn     PORTD
#define PORT_SH_RSTn      PORTD
#define PORT_CAMERA_LATCH PORTB

#define BIT_SH_CLK_UV     2
#define BIT_SH_DIN_UV     3
#define BIT_SH_CLK_VIS    0
#define BIT_SH_DIN_VIS    4
#define BIT_LED_LATCH     5
#define BIT_OUT_RSTn      6
#define BIT_SH_RSTn       7
#define BIT_CAMERA_LATCH  5

//power supply enable, power-good
#define PIN_VS_UV_EN    11 //PB3
#define PIN_VS_VIS_EN   10 //PB2

#define PIN_VS_UV_PG    12 //PB4
#define PIN_VS_VIS_PG   9 //PB1

//power supply loopback, divided by 10
#define PIN_5V0_10_ANA    A1
#define PIN_VS_VIS_10_ANA A6
#define PIN_VS_UV_10_ANA  A7

//current sense for each supply
#define PIN_5V0_CURRENT_ANA     A0
#define PIN_VS_UV_CURRENT_ANA   A2
#define PIN_VS_VIS_CURRENT_ANA  A3

//reserve for I2C
#define PIN_I2C_SDA   A4 
#define PIN_I2C_SCL   A5

//DAC outputs to set the LED currents
#define DAC_CH_VSET_UV 0  //output A
#define DAC_CH_VSET_VIS 1 //output B

#endif //LED_CONTROLLER_PINS_H
