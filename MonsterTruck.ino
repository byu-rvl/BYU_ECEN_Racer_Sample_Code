#include <Servo.h>
#include "Pitches.h"

//************** CONTROL SYSTEM VARS ******************//
float speedInput = 0.0;                 // meters/s
float speedOutput = 0.0;
bool  wasItZero = true;
char  useControlSystem = 0;             // don't use PID
float prevError = 0;
float kd = 0.05;
float kp = 0.1;
float ki = 0.0;
#define TIRE_CIRCUMFERENCE 0.215        // meters (small car)
#define ENCODER_PULSES_PER_TIRE_REV 360 // This is for one revolution of the tire
#define PULSES_PER_METER 1675           // ENCODER_PULSES_PER_TIRE_REV/TIRE_CIRCUMFERENCE
#define MAX_TURN 30                     // Physically limitted to +/- 30 degrees
#define MIN_SPEED 0.5                   // Minimun speed in meters per second
#define MAX_SPEED 3.0                   // Maximum speed in meters per second
#define PWM_OFFSET_POS 75               // 1565 for 0.5 m/s
#define PWN_OFFSET_NEG 52               // 1448 for -0.5 m/s
#define PWM_PER_METER 18                // PWM value per one meter per second. It is actually not linear. 

//******************* SERIAL OBJECTS ******************//
String  response = "";
String  command = "";
float   value = 0;
#define BAUD_RATE 115200

//******************* SERVO OBJECTS ******************//
Servo steeringServo;
Servo powerServo;
int POWER_PIN = 5;
int STEERING_PIN = 9;
int NEUTRALS = 1500;               // Hard Right - 2000. Hard Left - 1000. Middle - 1500.
int NEUTRALP = 1500;               // Fast Forward - 2000. Fast Reverse - 1000. Stopped - 1500.
int PULSES = 1500;                 // Hard Right - 2000. Hard Left - 1000. Middle - 1500.
int PULSEP = 1500;                 // Fast Forward - 2000. Fast Reverse - 1000. Stopped - 1500.
bool FORWARD = true;
//******************* ENCODER OBJECTS ******************//
#define encoderPinA 2              // one of the two interrupt pins on Nano and Uno
#define updateEncoderPeriod 1000     // value in ms
unsigned long encoderPos = 0;
float encoderChange = 0;
signed long encoderLast = 0;
unsigned long startTime = 0;
unsigned long endTime = 0;
int encoderPrev = HIGH;

//******************* RADIO CONTROL *****************//
#define RADIO_INPUT 0
#define SERIAL_INPUT 1
int RADIO_POWER_PIN = 11;
int RADIO_STEERING_PIN = 6;
int wait_timer = 0;
int mode = SERIAL_INPUT;
int NOTE_PIN = 12;
const int maxNotes = 8;
const int maxMelodies = 9;
int noteIndex = 0;
int melodyIndex = 0;
bool playMelody = false;

// notes in the melody:
int melody[maxMelodies][maxNotes] = { 
    {NOTE_C4, NOTE_G3, NOTE_G3, NOTE_A3, NOTE_G3, 0, NOTE_B3, NOTE_C4},
    {NOTE_E4, NOTE_C4, NOTE_D4, NOTE_G3, NOTE_G3, NOTE_D4, NOTE_E4, NOTE_C4},       // School Zone   
    {NOTE_C1, NOTE_C1, NOTE_C1, NOTE_C1, NOTE_C1, NOTE_C1, NOTE_C1, NOTE_C1},       // Construction (number changes how low or high,  0 is the lowest)
    {NOTE_E5, NOTE_C5, NOTE_E5, NOTE_C5, NOTE_E5, NOTE_C5, NOTE_E5, NOTE_C5},       // Railroad (number changes how low or high,  0 is the lowest)
    {NOTE_D4, NOTE_CS4, NOTE_D4, NOTE_CS4, NOTE_D4, NOTE_AS3, NOTE_G3, NOTE_F3},    // BYU Fight Song
    {NOTE_G3, NOTE_G3, NOTE_A3, NOTE_G3, NOTE_E3, NOTE_C4, NOTE_A3, NOTE_G3},       //    
    {NOTE_A3, NOTE_A3, NOTE_A3, NOTE_A3, NOTE_A3, NOTE_A3, NOTE_A3, NOTE_A3},
    {NOTE_C3, NOTE_E3, NOTE_G3, NOTE_C4, NOTE_G3, NOTE_E3, NOTE_C3, 0},
    {NOTE_G4, NOTE_GS4, NOTE_A4, NOTE_AS4, NOTE_B4, NOTE_C5, NOTE_CS5, NOTE_D5},
};
// note durations: 4 = quarter note, 8 = eighth note, etc.:
int noteDurations[maxMelodies][maxNotes] = {
    {4, 8, 8, 4, 4, 4, 4, 4},
    {2, 2, 2, 1, 2, 2, 2, 1},
    {8, 8, 8, 8, 8, 8, 8, 8},
    {4, 4, 4, 4, 4, 4, 4, 4},
    {2, 4, 2, 4, 4, 4, 4, 4},
    {8, 8, 4, 8, 4, 4, 4, 2},
    {4, 2, 4, 4, 4, 2, 4, 4},
    {4, 4, 4, 4, 4, 4, 4, 4},
    {4, 4, 4, 4, 4, 4, 4, 4},
};
void play()
{
    int noteDuration = 1000 / noteDurations[melodyIndex][noteIndex];
    tone(NOTE_PIN, melody[melodyIndex][noteIndex], noteDuration);
    //int pauseBetweenNotes = noteDuration * 1.30;
    delay(noteDuration);
    noteIndex++;
    if (noteIndex == maxNotes) playMelody = false;
}

void setup()
{
  pinMode(STEERING_PIN, OUTPUT);
  pinMode(POWER_PIN, OUTPUT);
  Serial.begin(BAUD_RATE);                  // opens serial port, sets baud rate
  steeringServo.attach(STEERING_PIN);
  steeringServo.writeMicroseconds(PULSES);
  
  powerServo.attach(POWER_PIN);
  powerServo.writeMicroseconds(PULSEP);
  pinMode(encoderPinA, INPUT);
  attachInterrupt(0, doEncoderA, RISING);
  
  startTime = millis();
  endTime = startTime+updateEncoderPeriod;
}

int checkForSerial(){     
  //Function that checks if serial is available and parses commands
  if (Serial.available() > 0) {
    int inChar = Serial.read(); 
    if (isDigit(inChar) || inChar == '.' || inChar == '-') {
      response += (char)inChar;
    } else if(isAlpha(inChar)) { 
      command += (char)inChar;
    } else if (inChar == '\n') { 
      // Serial.println(command);
      // Serial.println(response);       
        if(command=="Steer" || command=="steer") {         // steering wheel angle -30 degrees ~ 30 degrees
            PULSES = convertDegreesToPulse(response.toFloat());
            //Serial.print("Steering: ");
            //Serial.println(PULSES, DEC);
        } else if(command=="Drive" || command=="drive") {  // speed in meters per second, can be positive or negative
            value = response.toFloat();
            value = (value < -MAX_SPEED) ? -MAX_SPEED : value;
            value = (value > MAX_SPEED) ? MAX_SPEED: value;
            speedInput = value;
            if (speedInput == 0) wasItZero = true;
            // speedOutput = (speedOutput == 0 || speedInput == 0) ? value : speedOutput;
            if(useControlSystem == 0) {       // Don't use PID
                PULSEP = convertSpeedToPulse(value);
                //Serial.print("Speed: ");
                //Serial.println(PULSEP, DEC);
            }  
        } else if(command=="music") {               // play melody
            playMelody = true; 
            noteIndex = 0;
            melodyIndex = (int) response.toInt();
            Serial.println("Music");
            Serial.println(melodyIndex);
        } else if(command=="Zero" || command=="zero"){    // set neutral value (going straight) and turn the front wheels straight
          value = response.toFloat();
          if(value >= 1000 && value <= 2000) {
              NEUTRALS = int(value);
              PULSES = NEUTRALS;
              //Serial.print("Straight 0: ");
              //Serial.println(PULSES, DEC);
           }else{
              //Serial.println("Value not acceptable for command: straight");
          }
        } else if (command == "Encoder" || command == "encoder") {  
            Serial.println(encoderPos);  
        } else if(command=="KP" || command=="kp") {              // set kp value
            if (response.toFloat() >= 0.0 && response.toFloat() <= 1.0) {
                kp = response.toFloat();            
                //Serial.print("kp : ");
                //Serial.println(value, DEC);
            }
        } else if(command=="KD" || command=="kd") {              // set kd value
            if (response.toFloat() >= 0.0 && response.toFloat() <= 1.0) {
                kd = response.toFloat();             
                //Serial.print("kd : ");
                //Serial.println(value, DEC);
            }
        } else if (command=="Pid" || command=="pid") {                // 1: use PID, 0: don't use PID
            if (response.toInt() == 0) {
                useControlSystem = 0;
              //Serial.println("Not using control system");
            } else {
                useControlSystem = 1;
                //Serial.println("Using control system");
            }
        }
        // clear the string for new input:
        response = "";
        command = "";
      }
    }  
    return 0;
}

void doEncoderA(){                         // ISR to monitor interrupt Pin# 2 for encoder signal A
    encoderPos = encoderPos + 1;         // Only one direction so always +1
    //Serial.print("Encoder: ");
    //Serial.println(encoderPos);
}

int convertDegreesToPulse(float carAngle){
    carAngle = (carAngle < -MAX_TURN) ? -MAX_TURN : carAngle;
    carAngle = (carAngle > MAX_TURN) ? MAX_TURN: carAngle;
    int pulse = (int)(NEUTRALS + carAngle * 500 / MAX_TURN);
    if(pulse >= 2000) pulse = 2000;
    if(pulse <= 1000) pulse = 1000;
    return pulse;
}

int convertSpeedToPulse(float speedSetting){    // speed is in meters per second
  int pulse;
  if (speedSetting == 0) {
      pulse = NEUTRALP;
  } else if (speedSetting > 0) {
      if (FORWARD == false) {
        powerServo.writeMicroseconds(NEUTRALP);
        delay(100);
      }
      pulse = (int)((speedSetting-MIN_SPEED) * PWM_PER_METER + PWM_OFFSET_POS + NEUTRALP);
      FORWARD = true;
  } else {
      pulse = (int)((speedSetting+MIN_SPEED) * PWM_PER_METER - PWN_OFFSET_NEG + NEUTRALP);  
      if (FORWARD == true) {
         powerServo.writeMicroseconds(NEUTRALP);
         delay(100);
         powerServo.writeMicroseconds(1000);
         delay(100);
         powerServo.writeMicroseconds(NEUTRALP);
         delay(100);
         FORWARD = false;
      }
  }

  // Serial.print("Speed: ");
  // Serial.println (speedSetting, DEC);
  // Serial.print("Calculated PulseP: ");
  // Serial.println (pulse, DEC);
  return pulse;
}

void readDistance(){
    float distance = (float)encoderPos/PULSES_PER_METER;     // meters
    //Serial.print("D: ");
    //Serial.println (distance, DEC);
}

void updateEncoder(){
    startTime = millis();
    if(startTime >= endTime){
      encoderChange = encoderPos-encoderLast;
      if (encoderChange == 0) encoderPos = 0;      // reset encoder position when the car stops
      encoderLast = encoderPos;
      //Serial.println (encoderChange, DEC);
      endTime = startTime+updateEncoderPeriod; 
      if(useControlSystem==1) controlSystem();
    }
}

void controlSystem(){
  // function that will use the encoder frequency as the input to a PID control system
  // that will reach the desired speed.
  float actualSpeed = (encoderChange * (1000/updateEncoderPeriod))/PULSES_PER_METER;
  // switch actual speed sign if negative speed
  if (speedInput < 0) actualSpeed = -actualSpeed;
  //Calculate proportional error
  float error = speedInput - actualSpeed; 
  if (wasItZero == true && speedInput != 0) {
    wasItZero = false;
    speedOutput = speedInput;
    // Serial.println("Was zero");       
  } else if (encoderChange == 0) {
    speedOutput = 0;
  } else {
    speedOutput = speedOutput + error*kp + kd*(error-prevError);
  }
  prevError =  error;
  PULSEP = convertSpeedToPulse(speedOutput);         // It may have to go over MAX_SPEED to get the desired speed so don't limit speedOutput 
  if (speedOutput == 0 || speedInput == 0) PULSEP = NEUTRALP;
   //Serial.print("Output Speed: ");
   //Serial.println (speedOutput, DEC);
   //Serial.print("Actual Speed: ");
   //Serial.println (actualSpeed, DEC);
   //Serial.print("PulseP: ");
   //Serial.println (PULSEP, DEC);
}

void steeringControl(){
  if (mode == RADIO_INPUT) PULSES = pulseIn(RADIO_STEERING_PIN, HIGH);
  steeringServo.writeMicroseconds(PULSES);
}

void powerControl(){
  if (mode == RADIO_INPUT) PULSEP = pulseIn(RADIO_POWER_PIN, HIGH);
  powerServo.writeMicroseconds(PULSEP);
}

bool inRange(int val, int minimum, int maximum){
  return ((minimum <= val) && (val <= maximum));
}

void toggleMode(){
    wait_timer = 0;
    // toggle input mode when radio speed is very slow and radio turn is full left or full right
    while (!inRange(pulseIn(RADIO_STEERING_PIN, HIGH), 1100, 1900) && inRange(pulseIn(RADIO_POWER_PIN, HIGH), 1470, 1530)){
      steeringServo.writeMicroseconds(NEUTRALS);
      powerServo.writeMicroseconds(NEUTRALP);
      //Serial.print("Toggling ");
      delay(1000);
      wait_timer++;
      if (wait_timer == 1) {
        mode = 1 - mode;        // toggle between serial (1) and radio (0)
        PULSEP = NEUTRALP;
        PULSES = NEUTRALS;
        break;
      }
    }
    while (!inRange(pulseIn(RADIO_STEERING_PIN, HIGH), 1100, 1900) && inRange(pulseIn(RADIO_POWER_PIN, HIGH), 1470, 1530)){}
}

void loop()
{
  checkForSerial();
  updateEncoder();
  //toggleMode();
  steeringControl();
  powerControl();
  if (playMelody) play();
  //Serial.print("Steering: ");
  //Serial.println (pulseIn(RADIO_STEERING_PIN, HIGH), DEC);
  //Serial.print("Power: ");
  //Serial.println (pulseIn(RADIO_POWER_PIN, HIGH), DEC);
  //delay(200);
}
