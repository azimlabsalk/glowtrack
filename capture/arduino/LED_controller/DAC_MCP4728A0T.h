/* Drivers for the MCP4728A0T quad-output DAC. 
 *  Written by Mark Stambaugh (mstambaugh@UCSD.edu).
 *  Last edited 2021/06/29
 */

#ifndef MCP4728A0T_DAC_H
#define MCP4728A0T_DAC_H

#define MCP4728A0T_DAC_SLAVE_ADDRESS 0B01100000
#define MCP4728A0T_DAC_BITS 12

#include "Arduino.h"
#include <Wire.h>

class MCP4728A0T_DAC{
  public: 
    MCP4728A0T_DAC(uint16_t mV);
    
    bool test_comms();
    
    void write_ch_bits(byte ch, uint16_t val);
    void write_ch_mV(byte ch, uint16_t mV);

    void read_24_byte_register_map();
  
  private: 
    uint16_t _reference_mV;
    byte _ref_byte;
};


MCP4728A0T_DAC::MCP4728A0T_DAC(uint16_t mV){
  _reference_mV = mV;
  if(mV == 2048)
    _ref_byte = 0B10000000;
  else if(mV == 4096)
    _ref_byte = 0B10010000;
  else
    _ref_byte = 0B00000000;
}

bool MCP4728A0T_DAC::test_comms(){
  uint16_t data_read, data_write;

  //read the current setting from output 0
  Wire.requestFrom(MCP4728A0T_DAC_SLAVE_ADDRESS, 3);
  Wire.endTransmission();
  Wire.read(); //dummy read to clear 1st byte
  data_read = Wire.read();
  data_read = (data_read << 8) + Wire.read();
  data_read = data_read & 0x0FFF;

  //calculate new setting for output 0
  if(data_read == 0x0FFF)
    data_write = 0x0FFE;
  else
    data_write = (data_read + 1);

  //write the new setting to output 0
  write_ch_bits(0, data_write);

  //read the new setting from output 0 
  Wire.requestFrom(MCP4728A0T_DAC_SLAVE_ADDRESS, 3);
  Wire.endTransmission();
  Wire.read(); //dummy read to clear 1st byte
  data_read = Wire.read();
  data_read = (data_read << 8) + Wire.read();
  data_read = data_read & 0x0FFF;
  
  //check to see if it matches
  if(data_read == data_write)
    return 1;
  else
    return 0;
}

//see datasheet page 41, figure 5-10. 
void MCP4728A0T_DAC::write_ch_bits(byte ch, uint16_t val){
  byte data;
  val &= 0x0FFF; //limit to 0-4095 
  ch &= 0x03; //limit 0-3
  
  Wire.beginTransmission(MCP4728A0T_DAC_SLAVE_ADDRESS);
  data = 0B01011000 + (ch << 1);
  Wire.write(data);
  data = _ref_byte + (val >> 8);
  Wire.write(data);
  data = byte(val);
  Wire.write(data);
  Wire.endTransmission();
}

void MCP4728A0T_DAC::write_ch_mV(byte ch, uint16_t mV){
  uint32_t val = mV;
  val = val*4096/_reference_mV;

  if(val > 4095)
    val = 4095;
  
  write_ch_bits(ch, uint16_t(val));
}

void MCP4728A0T_DAC::read_24_byte_register_map(){
  Wire.requestFrom(MCP4728A0T_DAC_SLAVE_ADDRESS, 24);
  Wire.endTransmission();
  while(Wire.available())
    Wire.read();
}

#endif //MCP4728A0T_DAC_H
