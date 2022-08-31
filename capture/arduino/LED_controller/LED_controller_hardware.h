/*  Low-level hardware drivers for 240-channel LED controller.
 *  Written by Mark Stambaugh (mstambaugh@ucsd.edu) for Salk Azim Lab.
 *  Last edited 2021/05/19
 */

#ifndef LED_CONTROLLER_HARDWARE_H
#define LED_CONTROLLER_HARDWARE_H

#include <Wire.h>

#include "LED_controller_pins.h"
#include "DAC_MCP4728A0T.h"
#include "LED_controller_config.h"

MCP4728A0T_DAC DAC(4096);

//power supply control & status

inline uint16_t read_5V0_mV(){  return uint16_t(uint32_t(analogRead(PIN_5V0_10_ANA))*10*5000/1024); }
inline uint16_t read_VS_UV_mV(){  return uint16_t(uint32_t(analogRead(PIN_VS_UV_10_ANA))*10*5000/1024);}
inline uint16_t read_VS_VIS_mV(){  return uint16_t(uint32_t(analogRead(PIN_VS_VIS_10_ANA))*10*5000/1024);}

inline uint16_t read_5V0_current_mA(){  return uint16_t(uint32_t(analogRead(PIN_5V0_CURRENT_ANA))*5000/1024);}
inline uint16_t read_VS_UV_current_mA(){  return uint16_t(uint32_t(analogRead(PIN_VS_UV_CURRENT_ANA))*5000/1024);}
inline uint16_t read_VS_VIS_current_mA(){  return uint16_t(uint32_t(analogRead(PIN_VS_VIS_CURRENT_ANA))*5000/1024);}

uint16_t read_VS_UV_current_mA_overample(uint16_t count);
uint16_t read_VS_VIS_current_mA_overample(uint16_t count);

inline void enable_VS_UV(bool en){  digitalWrite(PIN_VS_UV_EN, en);}
inline void enable_VS_VIS(bool en){  digitalWrite(PIN_VS_VIS_EN, en);}

inline bool read_VS_UV_PG(){  return digitalRead(PIN_VS_UV_PG);}
inline bool read_VS_VIS_PG(){  return digitalRead(PIN_VS_VIS_PG);}


//clears all LED and camera trigger registers but does not clear the outputs
inline void reset_shift_registers(){
//  digitalWrite(PIN_SH_RSTn, LOW);
//  digitalWrite(PIN_SH_RSTn, HIGH);
  PORT_SH_RSTn &= ~(1<<BIT_SH_RSTn);
  PORT_SH_RSTn |=   1<<BIT_SH_RSTn;
}

//clears all LED enables and camera trigger outputs but does not clear the registers
inline void reset_shift_register_outputs(){
//  digitalWrite(PIN_OUT_RSTn, LOW);
//  digitalWrite(PIN_OUT_RSTn, HIGH);
  PORT_OUT_RSTn &= ~(1<<BIT_OUT_RSTn);
  PORT_OUT_RSTn |=   1<<BIT_OUT_RSTn;
}

//programs the camera trigger outputs with whatever is in the register
inline void latch_camera(){
//  digitalWrite(PIN_CAMERA_LATCH, HIGH);
//  digitalWrite(PIN_CAMERA_LATCH, LOW);
  PORT_CAMERA_LATCH |=   1<<BIT_CAMERA_LATCH;
  PORT_CAMERA_LATCH &= ~(1<<BIT_CAMERA_LATCH);
}

//programs the LED enable outputs with whatever is in the register
inline void latch_LEDs(){
//  digitalWrite(PIN_LED_LATCH, HIGH);
//  digitalWrite(PIN_LED_LATCH, LOW);
  PORT_LED_LATCH |=   1<<BIT_LED_LATCH;
  PORT_LED_LATCH &= ~(1<<BIT_LED_LATCH);
}

//shifts out one byte of data onto the UV register chain. Line is shared with the camera. 
inline void program_UV_reg(byte data){
  byte i;
  for(i=0; i<8; i++){
//    digitalWrite(PIN_SH_DIN_UV, data >> 7);
//    digitalWrite(PIN_SH_CLK_UV, HIGH);
//    digitalWrite(PIN_SH_CLK_UV, LOW);
    
    if(data>>7)
      PORT_SH_DIN_UV |= 1<<BIT_SH_DIN_UV;
    else
      PORT_SH_DIN_UV &= ~(1<<BIT_SH_DIN_UV);
    PORT_SH_CLK_UV |=   1<<BIT_SH_CLK_UV;
    PORT_SH_CLK_UV &= ~(1<<BIT_SH_CLK_UV);
    
    data = data << 1;
  }
}

//shifts out one byte of data onto the VIS register chain. 
inline void program_VIS_reg(byte data){
  byte i;
  for(i=0; i<8; i++){
//    digitalWrite(PIN_SH_DIN_VIS, data >> 7);
//    digitalWrite(PIN_SH_CLK_VIS, HIGH);
//    digitalWrite(PIN_SH_CLK_VIS, LOW);

    if(data>>7)
      PORT_SH_DIN_VIS |= 1<<BIT_SH_DIN_VIS;
    else
      PORT_SH_DIN_VIS &= ~(1<<BIT_SH_DIN_VIS);
    PORT_SH_CLK_VIS |=   1<<BIT_SH_CLK_VIS;
    PORT_SH_CLK_VIS &= ~(1<<BIT_SH_CLK_VIS);
    
    data = data << 1;
  }
}

//shifts out one byte of data into the camera. This is the first register on the UV register chain. 
inline void program_camera_reg(byte data){
  program_UV_reg(data);
}

inline void set_current_UV_mA(uint16_t mA){  
  DAC.write_ch_mV(DAC_CH_VSET_UV, mA); //1mA = 1mV 
}

inline void set_current_VIS_mA(uint16_t mA){  
  DAC.write_ch_mV(DAC_CH_VSET_VIS, mA); //1mA = 1mV 
}

void load_all_parameters(Parameters *p){
  //set the currents with the DAC
  set_current_UV_mA(p->UV_mA);
  delay(100); //TODO is this long delay necessary? Would 10msec work? The DAC definitely needs some delay. 
  set_current_VIS_mA(p->VIS_mA);

  byte i;
  //load the VIS registers
  for(i=20; i>0; i--){
    program_VIS_reg(p->VIS_en[i-1]);
  }
  
  //load the UV registers
  for(i=20; i>0; i--){
    program_UV_reg(p->UV_en[i-1]);
  }
  
  //load the camera register. Must be done last after all UV registers because they share data & clock lines. 
  program_camera_reg(p->camera_en); 
}

inline void program_VIS_registers(byte en[20]){
  byte i;
  //load the VIS registers
  for(i=20; i>0; i--){
    program_VIS_reg(en[i-1]);
  }
}

inline void program_UV_registers(byte en[20]){
  byte i;
  
  //load the UV registers
  for(i=20; i>0; i--){
    program_UV_reg(en[i-1]);
  }
}

uint16_t read_VS_UV_current_mA_overample(uint16_t count){
  uint16_t i;
  uint32_t total = 0;
  for(i=0; i<count; i++){
    total += analogRead(PIN_VS_UV_CURRENT_ANA);
  }
  total = total/count;
  return uint16_t(total*5000/1024);
}

uint16_t read_VS_VIS_current_mA_overample(uint16_t count){
  uint16_t i;
  uint32_t total = 0;
  for(i=0; i<count; i++){
    total += analogRead(PIN_VS_VIS_CURRENT_ANA);
  }
  total = total/count;
  return uint16_t(total*5000/1024);
}

#endif //LED_CONTROLLER_HARDWARE_H
