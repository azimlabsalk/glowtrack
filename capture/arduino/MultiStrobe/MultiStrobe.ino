/*

Code for strobing lights and triggering cameras in sequence. 

The Arduino must be connected to a computer via the USB serial port.
Commands are sent from the PC to the Arduino over USB serial as strings of ASCII characters.

There are 5 commands:

start - start triggering
stop - stop triggering
setduration:<float> - sets the duration of one cycle, in millis; defaults to 10.0
settime:<int>:<float> - sets the time for the trigger event at index <int> to value <float>; only needed for dynamically adjusting the trigger timing
setmode:<int> - set mode to <int>; mode of -1 is the default, and the active visible LED cycles; for all other values, the active visible LED remains at index 0

*/

#include "MultiStrobe.h"

#define BAUD_RATE 9600
 
#define UV_PIN 10   // controls UV LEDs
#define VIS_PIN 9   // controls visible LEDs - this is a dummy value - the actual pin is vis_pins[vis_idx] (see below)
#define CAM1_PIN 7  // controls monochrome cameras
#define CAM2_PIN 11 // controls color cameras
#define LED_PIN 13  // indicator LED

// Variables will change :
bool isRunning = false;

unsigned long cycleStart;              // time when the current control cycle started (in microseconds)
unsigned long cycleDuration = 10000;   // duration of one control cycle (in microseconds)

#define NUM_TRIGGERS 12

trigger triggers[] =
//pin, value, time
{
  { UV_PIN, HIGH, 0, false},
  { UV_PIN, LOW, 2500, false },
  { CAM1_PIN, HIGH, 2500, false },
  { CAM1_PIN, LOW, 4500, false },
  { CAM2_PIN, HIGH, 2500, false },
  { CAM2_PIN, LOW, 4500, false },
  { VIS_PIN, HIGH, 5000, false }, // VIS_PIN is a dummy value that means "turn on the active visible LED" (vis_pins[vis_idx])
  { VIS_PIN, LOW, 10000, false }, // VIS_PIN is a dummy value that means "turn on the active visible LED" (vis_pins[vis_idx])
  { CAM1_PIN, HIGH, 7500, false },
  { CAM1_PIN, LOW, 9500, false },
  { CAM2_PIN, HIGH, 7500, false },
  { CAM2_PIN, LOW, 9500, false },
};

#define N_VIS_PINS 9
int vis_pins[] = {40, 41, 42, 43, 30, 31, 32, 33, 34};  // these pins control the visible LEDs
int vis_idx = 0;                                        // the index of the active visible LED

#define VIS_REPETITIONS 1;  // number of cycles to trigger the same visible LED before vis_idx is incremented
int vis_rep_counter = 0;    // number of cycles that have elapsed since vis_idx was incremented

int mode = -1;

void setup() {

  // initialize pins

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  pinMode(UV_PIN, OUTPUT);
  pinMode(VIS_PIN, OUTPUT);
  pinMode(CAM1_PIN, OUTPUT);
  pinMode(CAM2_PIN, OUTPUT);
  digitalWrite(UV_PIN, LOW);
  digitalWrite(VIS_PIN, LOW);
  digitalWrite(CAM1_PIN, LOW);
  digitalWrite(CAM2_PIN, LOW);

  for (int i = 0; i < N_VIS_PINS; i++) {
    int pin = vis_pins[i];
    pinMode(pin, OUTPUT);
    digitalWrite(pin, LOW);
  }

  // set up USB communication link
  Serial.begin(BAUD_RATE);

  // initialize control cycle
  cycleStart = micros();

}

void loop() {

  bool run = isRunning; // this is intended to solve interrupt issues
  unsigned long currentTime = micros();

  if (run && mode == -1) {  // this is the default mode: the visible LEDs cycle

    unsigned long cycleTime = currentTime - cycleStart;

    for (int i = 0; i < NUM_TRIGGERS; i++) {
      if (!triggers[i].triggered && cycleTime > triggers[i].time) {
        triggers[i].triggered = true;
        if (triggers[i].pin == VIS_PIN) {  // we are setting the visible pin
          digitalWrite(vis_pins[vis_idx], triggers[i].value);
          digitalWrite(LED_PIN, triggers[i].value);
          if (triggers[i].value == LOW) {  // if visible pin just went low, a cycle just finished 
            vis_rep_counter = (vis_rep_counter + 1) % VIS_REPETITIONS;
            if (vis_rep_counter == 0) {
              vis_idx = (vis_idx + 1) % N_VIS_PINS;
            }
          }
        }
        else {
          digitalWrite(triggers[i].pin, triggers[i].value);
        }
      }
    }

    if (cycleTime > cycleDuration) {
      cycleStart = currentTime;
      for (int i = 0; i < NUM_TRIGGERS; i++) {
        triggers[i].triggered = false;
      }
    }

  }
  else if (run && mode != -1) {  // this is a debug mode - the first visible pin is always triggered

    unsigned long cycleTime = currentTime - cycleStart;

    vis_idx = mode;

    for (int i = 0; i < NUM_TRIGGERS; i++) {
      if (!triggers[i].triggered && cycleTime > triggers[i].time) {
        triggers[i].triggered = true;
        if (triggers[i].pin == VIS_PIN) {
          digitalWrite(vis_pins[vis_idx], triggers[i].value);
          digitalWrite(LED_PIN, triggers[i].value);
        }
        else {
          digitalWrite(triggers[i].pin, triggers[i].value);
        }
      }
    }

    if (cycleTime > cycleDuration) {
      cycleStart = currentTime;
      for (int i = 0; i < NUM_TRIGGERS; i++) {
        triggers[i].triggered = false;
      }
    }


  }
  else {
    cycleStart = currentTime;
  }

}

void start_running() {

  isRunning = true;

  cycleStart = micros();

  vis_rep_counter = 0;
  vis_idx = 0;

  digitalWrite(UV_PIN, LOW);
  digitalWrite(VIS_PIN, LOW);
  digitalWrite(CAM1_PIN, LOW);
  digitalWrite(CAM2_PIN, LOW);

}

void stop_running() {

  isRunning = false;
  
  cycleStart = micros();

  digitalWrite(UV_PIN, LOW);
  digitalWrite(VIS_PIN, LOW);
  digitalWrite(CAM1_PIN, LOW);
  digitalWrite(CAM2_PIN, LOW);

  for (int i = 0; i < N_VIS_PINS; i++) {
    int pin = vis_pins[i];
    pinMode(pin, OUTPUT);
    digitalWrite(pin, LOW);
  }

}

void set_trigger_time(int idx, float ms) {
  triggers[idx].time = 1000 * ms;
}

void set_cycle_duration(float ms) {
  cycleDuration = 1000 * ms;
}

void serialEvent() {
  String commandString, commandName;
  float commandValue;
  int commandIdx;

  while (Serial.available()) {
    commandString = Serial.readStringUntil('\n');
  }

  int idx = commandString.indexOf(':');
  if (idx == -1) {
    commandName = commandString;
  }
  else {
    commandName = commandString.substring(0, idx);
  }

  if (commandName == "start") {
    // 'start'
    start_running();
  }
  else if (commandName == "stop") {
    // 'stop'
    stop_running();
  }
  else if (commandName == "setduration") {
    // 'setduration:10.0'
    commandValue = commandString.substring(idx + 1).toFloat();
    set_cycle_duration(commandValue);
  }
  else if (commandName == "settime") {
    // 'settime:1:10.0'
    int idx2 = commandString.indexOf(':', idx + 1);
    commandIdx = commandString.substring(idx + 1, idx2).toInt();
    commandValue = commandString.substring(idx2 + 1).toFloat();
    set_trigger_time(commandIdx, commandValue);
  }
  else if (commandName == "setmode") {
    // 'setmode:0'
    commandValue = commandString.substring(idx + 1).toInt();
    boolean wasRunning = isRunning;
    if (wasRunning) { stop_running(); }
    mode = commandValue;
    if (wasRunning) { start_running(); }
  }
}
