#include <Arduino.h>

typedef struct
{
  int pin;
  int value;
  unsigned long time;
  bool triggered;
} trigger;
