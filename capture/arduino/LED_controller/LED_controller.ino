/*  Controller for a 240-ch LED array. 
 *  Written by Mark Stambaugh (mstambaugh@ucsd.edu) for Salk Azim Lab. 
 *  Last edited 2021/08/09
 */


//NOTE the camera trigger shift register shares SH_DIN_UV and SH_CLK_UV. It is the first shift register in that chain. 

#include "LED_controller_hardware.h"
#include "LED_controller_config.h"

#ifndef PRINT
#define PRINT(...)  if (!quiet) Serial.print(__VA_ARGS__)
#endif

#ifndef PRINTLN
#define PRINTLN(...) if (!quiet) Serial.println(__VA_ARGS__)
#endif

Parameters params_run;

bool quiet = true;

void set_parameters();
void get_parameters();
void run_trial();
void enable_power();
void disable_power();
void test_daughtercard();
void check_for_LEDs();
void print_help();

void go_quiet(){
  quiet = true; PRINTLN('q');
}

void go_verbose(){
  quiet = false; PRINTLN("verbose mode.");
}

//void PRINTLN(byte* s) {
//  Serial.PRINTLN(s);
//}
//
//void PRINT(byte* s) {
//  Serial.PRINT(s);
//}

inline void clear_serial_buffer(){ 
  while(Serial.available()) 
    Serial.read(); 
}
void skip_serial_whitespace(){ 
  while((Serial.peek() == '\n') || (Serial.peek() == '\r') || (Serial.peek() == '\t') || (Serial.peek() == ' '))
    Serial.read();
}

void setup(){
  //port directions
  pinMode(PIN_SH_CLK_UV, OUTPUT);
  pinMode(PIN_SH_DIN_UV, OUTPUT); 
  
  pinMode(PIN_SH_CLK_VIS, OUTPUT); 
  pinMode(PIN_SH_DIN_VIS, OUTPUT); 
  
  pinMode(PIN_LED_LATCH, OUTPUT); 
  pinMode(PIN_OUT_RSTn, OUTPUT); 
  pinMode(PIN_SH_RSTn, OUTPUT); 
  pinMode(PIN_CAMERA_LATCH, OUTPUT); 
  
  pinMode(PIN_VS_UV_EN, OUTPUT); 
  pinMode(PIN_VS_VIS_EN, OUTPUT); 
  
  pinMode(PIN_VS_UV_PG, INPUT); 
  pinMode(PIN_VS_VIS_PG, INPUT); 
  
  pinMode(PIN_5V0_10_ANA, INPUT); 
  pinMode(PIN_VS_VIS_10_ANA, INPUT); 
  pinMode(PIN_VS_UV_10_ANA, INPUT); 
  
  pinMode(PIN_5V0_CURRENT_ANA, INPUT); 
  pinMode(PIN_VS_UV_CURRENT_ANA, INPUT); 
  pinMode(PIN_VS_VIS_CURRENT_ANA, INPUT); 

  //set initial values
  digitalWrite(PIN_SH_CLK_UV, LOW);
  digitalWrite(PIN_SH_DIN_UV, LOW);
    
  digitalWrite(PIN_SH_CLK_VIS, LOW);
  digitalWrite(PIN_SH_DIN_VIS, LOW);
    
  digitalWrite(PIN_LED_LATCH, LOW);
  digitalWrite(PIN_OUT_RSTn, LOW);
  digitalWrite(PIN_SH_RSTn, LOW);
  digitalWrite(PIN_CAMERA_LATCH, LOW);

  digitalWrite(PIN_VS_UV_EN, LOW);
  digitalWrite(PIN_VS_VIS_EN, LOW);


  Serial.begin(115200);
  Serial.setTimeout(10);
  delay(10);
  
  PRINTLN();
  PRINTLN(F("Running LED controller startup."));
  
  //clear existing registers
  reset_shift_registers();
  reset_shift_register_outputs();

  //load default parameters
  params_run = params_default;
  load_all_parameters(&params_run);
  
  Wire.begin();

  //test DAC, wait for user input upon failure. 
  while(1){
    if(!DAC.test_comms()){
      clear_serial_buffer();
      PRINTLN(F("DAC test failed. Enter 'i' to ignore or any other key to retry."));
      while(!Serial.available()){}
      if(Serial.peek() == 'i'){
        PRINTLN("Ignoring DAC failure. Vaya con Dios.");
        break;
      }
    }
    else{
      PRINTLN(F("DAC test good."));
      break;
    }
  }

  clear_serial_buffer();
  
  PRINTLN(F("Startup complete."));
  PRINTLN(F("Send 'h' for help."));
  PRINTLN();
}

void loop() {
  char cmd;
  
  while(!Serial.available()){}
  
  skip_serial_whitespace();
  
  while(Serial.available()){
    cmd = Serial.read();
    switch(cmd){
      case 's': set_parameters(); break;
      case 'g': get_parameters(); break;
      case 'r': run_trial(); break;
      case 'f': run_free(); break;
      case 'e': enable_power(); break;
      case 'd': disable_power(); break;
      case 'q': go_quiet(); break;
      case 'v': go_verbose(); break;
      case 't': test_daughtercard(); break;
      case 'c': check_for_LEDs(); break;
      case 'h': print_help(); break;
      default: 
        PRINT(F("ERROR: Invalid command: ")); 
        PRINTLN(cmd); 
        clear_serial_buffer();
        break;
    }
    skip_serial_whitespace();
  }
}

void print_help(){
  PRINTLN(F("Commands ARE case sensitive."));
  PRINTLN(F("h: print help menu."));
  PRINTLN(F("q: enter quiet mode."));
  PRINTLN(F("v: enter verbose mode."));
  PRINTLN(F("e: enable power."));
  PRINTLN(F("d: disable power."));
  PRINTLN(F("c: check all cards for installed LEDs."));
  PRINTLN(F("t + u/v + #: test UV/visible daughtercard #. Ex: tv1 to test visible #1."));
  PRINTLN(F("s + <data>: set parameters."));
  PRINTLN(F("g: get current parameters."));
  PRINTLN(F("r: run trial with current parameters."));
  PRINTLN(F("f: run freely until receiving any character."));
  PRINTLN();  
}

void set_parameters(){
  byte i;

  params_run.UV_mA = Serial.parseInt();
  params_run.VIS_mA = Serial.parseInt();
  
  for(i=0; i<20; i++){
    params_run.UV_en[i] = Serial.parseInt();
  }
  
  for(i=0; i<20; i++){
    params_run.VIS_en[i] = Serial.parseInt();
  }
  
  params_run.camera_en = Serial.parseInt();
  
  params_run.camera_VIS_on_us = Serial.parseInt();
  params_run.LED_VIS_off_us = Serial.parseInt();
  params_run.LED_UV_on_us = Serial.parseInt();
  params_run.LED_UV_off_us = Serial.parseInt();
  params_run.camera_UV_on_us = Serial.parseInt();
  
  params_run.frame_length_us = Serial.parseInt();
  params_run.frame_count = Serial.parseInt();
  
  bool valid = true;
  
  //check parameters
  uint16_t LED_sum = 0;
  for(i=0; i<20; i++){
    LED_sum += params_run.UV_en[i];
    LED_sum += params_run.VIS_en[i];
  }
  
  if(LED_sum == 0){
    PRINTLN(F("ERROR: Invalid LED setting: all off."));
    valid = false;
  }
  if(params_run.camera_en == 0){
    PRINTLN(F("ERROR: Invalid camera setting: all off."));
    valid = false;
  }
  if((params_run.UV_mA > UV_MA_MAX) | (params_run.UV_mA < UV_MA_MIN)){
    PRINT(F("ERROR: Invalid UV current (mA): "));
    PRINTLN(params_run.UV_mA);
    valid = false;
  }
  if((params_run.VIS_mA > VIS_MA_MAX) | (params_run.VIS_mA < VIS_MA_MIN)){
    PRINT(F("ERROR: Invalid VIS current (mA): "));
    PRINTLN(params_run.VIS_mA);
    valid = false;
  }
  
  if(params_run.LED_VIS_off_us < params_run.camera_VIS_on_us){
    PRINTLN(F("ERROR: camera_VIS_on_us is after LED_VIS_off_us."));
    valid = false;
  }
  if(params_run.LED_UV_on_us < params_run.LED_VIS_off_us){
    PRINTLN(F("ERROR: LED_VIS_off_us is after LED_UV_on_us."));
    valid = false;
  }
  if(params_run.LED_UV_off_us < params_run.LED_UV_on_us){
    PRINTLN(F("ERROR: LED_UV_on_us is after LED_UV_off_us."));
    valid = false;
  }
  if(params_run.camera_UV_on_us < params_run.LED_UV_off_us){
    PRINTLN(F("ERROR: LED_UV_off_us is after camera_UV_on_us."));
    valid = false;
  }
  if(params_run.frame_length_us < params_run.camera_UV_on_us){
    PRINTLN(F("ERROR: camera_UV_on_us is after frame_length_us."));
    valid = false;
  }

  if((params_run.frame_length_us > FRAME_LENGTH_US_MAX) | (params_run.frame_length_us < FRAME_LENGTH_US_MIN)){
    PRINT(F("ERROR: Invalid frame length (us): "));
    PRINTLN(params_run.frame_length_us);
    valid = false;
  }
  if((params_run.frame_count > FRAME_COUNT_MAX) | (params_run.frame_count < FRAME_COUNT_MIN)){
    PRINT(F("ERROR: Invalid frame count: "));
    PRINTLN(params_run.frame_count);
    valid = false;
  }
  
  if(!valid){
    PRINTLN(F("ERROR: Invalid parameters. Loading default parameters."));
    params_run = params_default;
  }
  
  load_all_parameters(&params_run);
  
  if(!quiet)
    PRINTLN(F("Parameters set."));
  else
    PRINT('s');

  Serial.println();
}

void get_parameters(){
  if(quiet){ //TODO
    Serial.write((byte*)&params_run,53);
  }
  else{
    PRINTLN(F("Parameters:"));
    
    PRINT(F("UV current (mA): "));
    PRINTLN(params_run.UV_mA);
    PRINT(F("VIS current (mA): "));
    PRINTLN(params_run.VIS_mA);

    PRINTLN();
    for(byte i=0; i<20; i++){
      PRINT(F("LED UV "));
      PRINT(i+1);
      PRINT(F(": 0x"));
      PRINTLN(params_run.UV_en[i],HEX);
    }
    PRINTLN();
    for(byte i=0; i<20; i++){
      PRINT(F("LED VIS "));
      PRINT(i+1);
      PRINT(F(": 0x"));
      PRINTLN(params_run.VIS_en[i],HEX);
    }
    
    PRINT(F("cameras: 0x"));
    PRINTLN(params_run.camera_en, HEX);
    PRINT(F("camera VIS on (us): "));
    PRINTLN(params_run.camera_VIS_on_us);
    PRINT(F("LED VIS off (us): "));
    PRINTLN(params_run.LED_VIS_off_us);
    PRINT(F("LED UV on (us): "));
    PRINTLN(params_run.LED_UV_on_us);
    PRINT(F("LED UV off (us): "));
    PRINTLN(params_run.LED_UV_off_us);
    PRINT(F("camera UV on (us): "));
    PRINTLN(params_run.camera_UV_on_us);
    
    PRINT(F("frame length (us): "));
    PRINTLN(params_run.frame_length_us);
    PRINT(F("frame count: "));
    PRINTLN(params_run.frame_count);
  }
}

void test_daughtercard(){
  delay(10);
  char type = Serial.read();
  byte d = Serial.parseInt();
  if(d>20){
    PRINTLN(F("ERROR: Invalid daughtercard selected."));
    return;
  }
  
  if((type == 'u') || (type == 'U'))
    PRINT(F("Testing daughtercard UV #"));
  else if((type == 'v') || (type == 'V'))
    PRINT(F("Testing daughtercard VIS #"));
  else{
    PRINTLN(F("ABORT: Invalid arg."));
    return;
  }
  PRINTLN(d);
  
  params_test.UV_mA = UV_LED_TEST_CURRENT_MA;
  params_test.VIS_mA = VIS_LED_TEST_CURRENT_MA;
  
  //enable selected daughtercards LEDs 1 at a time
  
  PRINTLN("Send any character to end.");
  clear_serial_buffer();

  byte i;
  while(!Serial.available()){
    if((type == 'u') || (type == 'U'))
      params_test.UV_en[d-1] = 0x7E;
    else
      params_test.VIS_en[d-1] = 0x7E;
    load_all_parameters(&params_test);
    
    latch_LEDs();
    delay(TEST_DELAY_MS*2);
    reset_shift_register_outputs();
    if(Serial.available())
        break;
      
    for(i=1; i<=6; i++){
      if((type == 'u') || (type == 'U'))
        params_test.UV_en[d-1] = 0x01 << i;
      else
        params_test.VIS_en[d-1] = 0x01 << i;
      load_all_parameters(&params_test);
      
      latch_LEDs();
      delay(TEST_DELAY_MS);
      reset_shift_register_outputs();
      if(Serial.available())
        break;
    }
    delay(TEST_DELAY_MS*2);
  }
  clear_serial_buffer();
  
  //turn all LEDs off
  params_test.UV_en[d-1] = 0;
  params_test.VIS_en[d-1] = 0;
  
  reset_shift_register_outputs();
  PRINTLN(F("Test complete"));
}

void check_for_LEDs(){
  byte i, j;
  uint16_t idle_current_mA;
  uint16_t LED_current_mA;

  params_test.UV_mA = UV_LED_CHECK_CURRENT_MA;
  params_test.VIS_mA = VIS_LED_CHECK_CURRENT_MA;

  PRINTLN(F("Checking for LEDs. Enter any character to abort."));
  clear_serial_buffer();
  
  idle_current_mA = read_VS_VIS_current_mA_overample(100);
  PRINT("VIS idle current: ");
  PRINTLN(idle_current_mA);
  
  PRINTLN("VIS LEDs: ");
  PRINTLN("  123456");
  for(i=1; i<=CHECK_LED_VIS_CARD_COUNT; i++){
    PRINT(i);
    PRINT(' ');
    for(j=1; j<=6; j++){
      params_test.VIS_en[i-1] = (0x01 << j);
      load_all_parameters(&params_test);
      latch_LEDs();
      LED_current_mA = read_VS_VIS_current_mA_overample(CHECK_LED_OVERSAMPLE_COUNT);
      if((LED_current_mA - idle_current_mA) >= (VIS_LED_CHECK_CURRENT_MA/2))
        PRINT('X');
      else
        PRINT(' ');
      reset_shift_register_outputs();
    }
    params_test.VIS_en[i-1] = 0;
    PRINTLN();
    if(Serial.available()){
      PRINTLN("ABORT: serial received.");
      clear_serial_buffer();
      return;
    }
  }
  
  idle_current_mA = read_VS_UV_current_mA_overample(100);
  PRINT("UV idle current: ");
  PRINTLN(idle_current_mA);
  
  PRINTLN("UV LEDs: ");
  PRINTLN("  123456");
  for(i=1; i<=CHECK_LED_UV_CARD_COUNT; i++){
    PRINT(i);
    PRINT(' ');
    for(j=1; j<=6; j++){
      params_test.UV_en[i-1] = (0x01 << j);
      load_all_parameters(&params_test);
      latch_LEDs();
      LED_current_mA = read_VS_UV_current_mA_overample(CHECK_LED_OVERSAMPLE_COUNT);
      if((LED_current_mA - idle_current_mA) >= (UV_LED_CHECK_CURRENT_MA/2))
        PRINT('X');
      else
        PRINT(' ');
      reset_shift_register_outputs();
    }
    params_test.UV_en[i-1] = 0;
    PRINTLN();
    if(Serial.available()){
      PRINTLN("ABORT: serial received.");
      clear_serial_buffer();
      return;
    }
  }
}

/*
PHASE   0123456
VIS LED  XX    
UV  LED    X   
CAMERA       X 
*/


void run_trial(){
  //uint32_t frame_length_us = uint32_t(1000000)/params_run.frame_rate_Hz;
  
  uint32_t frame_start_us;
  uint32_t frame;

  set_current_UV_mA(params_run.UV_mA);
  delay(100); 
  set_current_VIS_mA(params_run.VIS_mA);
  delay(100);
  
  reset_shift_register_outputs();
  reset_shift_registers();
  
  //clear serial buffer
  delay(10);
  clear_serial_buffer();
  
  if(!quiet)
    PRINTLN(F("Running trial. Enter any character to abort."));
    
  PRINTLN(F("BEGIN"));
  
  //start the frame at the next integer second
  frame_start_us = micros();
  frame_start_us -= (frame_start_us%1000000);
  frame_start_us += 1000000;
  
  //wait for zeroeth frame to start. 
  while(micros() < frame_start_us);
  
  for(frame=params_run.frame_count; frame>0; frame--){
    //check to see if the trial should be aborted
    if(Serial.available()){
      PRINTLN(F("ABORT: Serial received."));
      clear_serial_buffer();
      break;
    }

    //PHASE 0: all LEDs off, no cameras capturing
    
    //load visible LED & camera data
    program_VIS_registers(&params_run.VIS_en[0]);
    program_camera_reg(&params_run.camera_en);
    
    //wait for previous frame to end. Visible LEDs & cameras trigger at start of frame. 
    while(micros() - frame_start_us < params_run.frame_length_us)
    {}
    
    //update the frame start time
    frame_start_us = frame_start_us + params_run.frame_length_us;

    //PHASE 1: visible LEDs on, no cameras capturing
    latch_LEDs();

    while(micros() - frame_start_us < params_run.camera_VIS_on_us)
    {}

    //PHASE 2: visible LEDs on, cameras capturing
    latch_camera();

    //wait for LED on-time to complete
    while(micros() - frame_start_us < params_run.LED_VIS_off_us)
    {}
    
    //reset registers & outputs and load UV LED & camera data
    reset_shift_register_outputs();
    reset_shift_registers();
    
    program_UV_registers(&params_run.UV_en[0]);
    program_camera_reg(&params_run.camera_en);
    
    //wait for UV start time
    while(micros() - frame_start_us < params_run.LED_UV_on_us)
    {}

    //PHASE 3: UV LEDs on, no cameras
    latch_LEDs();

    //wait for UV stop time
    while(micros() - frame_start_us < params_run.LED_UV_off_us)
    {}

    //PHASE 4: all LEDs off, no cameras capturing
    reset_shift_register_outputs();

    while(micros() - frame_start_us < params_run.camera_UV_on_us)
    {}

    //PHASE 5: all LEDS off, cameras capturing
    latch_camera();
    
    reset_shift_registers();
    reset_shift_register_outputs();
  }
  
  PRINTLN(F("END."));
}

// free-running collection
void run_free(){
  //uint32_t frame_length_us = uint32_t(1000000)/params_run.frame_rate_Hz;
  
  uint32_t frame_start_us;
  uint32_t frame;

  byte vis_idx = 0;
  byte n_vis = 9;

  set_current_UV_mA(params_run.UV_mA);
  delay(100); 
  set_current_VIS_mA(params_run.VIS_mA);
  delay(100);
  
  reset_shift_register_outputs();
  reset_shift_registers();
  
  //clear serial buffer
  delay(10);
  clear_serial_buffer();
  
  if(!quiet)
    PRINTLN(F("Running trial. Enter x character to abort."));
    
  PRINTLN(F("BEGIN"));

  Serial.println();

  //start the frame at the next integer second
  frame_start_us = micros();
  frame_start_us -= (frame_start_us%1000000);
  frame_start_us += 1000000;
  
  //wait for zeroeth frame to start. 
  while(micros() < frame_start_us);
  
//  for(frame=params_run.frame_count; frame>0; frame--) {

  while(true) {
    //check to see if the trial should be aborted
    if(Serial.available()){
      char cmd = Serial.read();
      if (cmd == 'x') {
        PRINTLN(F("ABORT: Serial received."));
        clear_serial_buffer();
        break;
      }
    }

    //PHASE 0: wait for previous frame to end

    //load visible LED & camera data
//    program_VIS_registers(&params_run.VIS_en[0]);
//    program_camera_reg(&params_run.camera_en);
    
    //wait for previous frame to end. Visible LEDs & cameras trigger at start of frame. 
    while(micros() - frame_start_us < params_run.frame_length_us)
    {}

    reset_shift_registers();
    reset_shift_register_outputs();

    //load visible LED & camera data
    program_VIS_registers_active(vis_idx);
    vis_idx = (vis_idx + 1) % n_vis;
    program_camera_reg(&params_run.camera_en);

    //update the frame start time
    frame_start_us = frame_start_us + params_run.frame_length_us;

    //PHASE 1: visible LEDs on, no cameras capturing
    latch_LEDs();

    while(micros() - frame_start_us < params_run.camera_VIS_on_us)
    {}

    //PHASE 2: visible LEDs on, cameras capturing
    latch_camera();

    //wait for LED on-time to complete
    while(micros() - frame_start_us < params_run.LED_VIS_off_us)
    {}
    
    //reset registers & outputs and load UV LED & camera data
    reset_shift_register_outputs();
    reset_shift_registers();
    
    program_UV_registers(&params_run.UV_en[0]);
    program_camera_reg(&params_run.camera_en);
    
    //wait for UV start time
    while(micros() - frame_start_us < params_run.LED_UV_on_us)
    {}

    //PHASE 3: UV LEDs on, no cameras
    latch_LEDs();

    //wait for UV stop time
    while(micros() - frame_start_us < params_run.LED_UV_off_us)
    {}

    //PHASE 4: all LEDs off, no cameras capturing
    reset_shift_register_outputs();

    while(micros() - frame_start_us < params_run.camera_UV_on_us)
    {}

    //PHASE 5: all LEDS off, cameras capturing
    latch_camera();

//    while(micros() - frame_start_us < params_run.camera_UV_on_us + 1000)
//    {}

//    reset_shift_registers();
//    reset_shift_register_outputs();
  }

  reset_shift_register_outputs();
  reset_shift_registers();
  
  PRINTLN(F("END."));

  Serial.println();

}


//start up the regulators one at a time, monitoring current and Power-Good signals. 
void enable_power(){
  uint32_t start_ms;
  bool startup_success;
  uint16_t current_mA;

  PRINTLN("Enabling regulators.");
  
  enable_VS_UV(true);
  startup_success = false;
  
  start_ms = millis();
  while(millis() - start_ms < VS_UV_STARTUP_TIMEOUT_MS){
    delay(100);
    current_mA = read_VS_UV_current_mA();
    //PRINTLN(current_mA);
    if((current_mA < VS_UV_IDLE_CURRENT_MA) & read_VS_UV_PG()){
      startup_success = true;
      break;
    }
  }
  
  if(startup_success){
    PRINTLN(F("UV supply up."));
  }
  else{
    PRINTLN(F("ERROR: UV supply startup timed out."));
  }
  
  enable_VS_VIS(true);
  startup_success = false;
  
  start_ms = millis();
  while(millis() - start_ms < VS_VIS_STARTUP_TIMEOUT_MS){
    delay(100);
    current_mA = read_VS_VIS_current_mA();
    //PRINTLN(current_mA);
    if((current_mA < VS_VIS_IDLE_CURRENT_MA) & read_VS_VIS_PG()){
      startup_success = true;
      break;
    }
  }
  
  if(startup_success){
    PRINTLN(F("VIS supply up."));
  }
  else{
    PRINTLN(F("ERROR: VIS supply startup timed out."));
  }
  
  PRINT(F("UV supply (mV):  "));
  PRINTLN(read_VS_UV_mV());
  PRINT(F("VIS supply (mV): "));
  PRINTLN(read_VS_VIS_mV());
  PRINT(F("5V0 supply (mV): "));
  PRINTLN(read_5V0_mV());

  Serial.println();

}


//shut down regulators one at a time monitoring output voltage
void disable_power(){
  uint32_t start_ms;
  bool shutdown_success;
  uint16_t voltage_mV;

  PRINTLN("Disabling regulators.");
  
  enable_VS_UV(false);
  enable_VS_VIS(false);
  shutdown_success = false;
  
  start_ms = millis();
  while(millis() - start_ms < VS_UV_SHUTDOWN_TIMEOUT_MS){
    delay(100);
    voltage_mV = read_VS_UV_mV();
    //PRINTLN(voltage_mV);
    if(voltage_mV < VS_UV_SHUTDOWN_MV){
      shutdown_success = true;
      break;
    }
  }
  
  if(shutdown_success){
    PRINTLN(F("UV supply down."));
  }
  else{
    PRINTLN(F("ERROR: UV supply shutdown timed out."));
  }
  
  enable_VS_VIS(false);
  shutdown_success = false;
  
  start_ms = millis();
  while(millis() - start_ms < VS_VIS_SHUTDOWN_TIMEOUT_MS){
    delay(100);
    voltage_mV = read_VS_VIS_mV();
    //PRINTLN(voltage_mV);
    if(voltage_mV < VS_VIS_SHUTDOWN_MV){
      shutdown_success = true;
      break;
    }
  }
  if(shutdown_success){
    PRINTLN(F("VIS supply down."));
  }
  else{
    PRINTLN(F("ERROR: VIS supply shutdown timed out."));
  }
}
