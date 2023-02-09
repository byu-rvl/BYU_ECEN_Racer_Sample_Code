/*

*/
int red1 = 2;
int yellow1 = 3;
int green1 = 4;
int red2 = 5;
int yellow2 = 6;
int green2 = 7;
int red3 = 8;
int yellow3 = 9;
int green3 = 10;

struct stopLight {
  int red;
  int yellow;
  int green;
};

stopLight light1 = {red1, yellow1, green1};
stopLight light2 = {red2, yellow2, green2};
stopLight light3 = {red3, yellow3, green3};

void setup(){
   pinMode(red1, OUTPUT);
   pinMode(yellow1, OUTPUT);
   pinMode(green1, OUTPUT);
   pinMode(red2, OUTPUT);
   pinMode(yellow2, OUTPUT);
   pinMode(green2, OUTPUT);
   pinMode(red3, OUTPUT);
   pinMode(yellow3, OUTPUT);
   pinMode(green3, OUTPUT);
}

void loop(){
  digitalWrite(light1.red, HIGH);
  digitalWrite(light2.red, HIGH);
  digitalWrite(light3.red, HIGH);
  delay(1000);      // Red ON

  lightCycle(light1);  
  delay(1000);
  lightCycle(light2); 
  delay(1000);
  lightCycle(light3); 
}

void lightCycle(stopLight light) {
  // Red off, green on
  digitalWrite(light.red, LOW);
  digitalWrite(light.green, HIGH);
  delay(5000);      // Green ON
  // Green off, yellow on
  digitalWrite(light.green, LOW);
  digitalWrite(light.yellow, HIGH);
  delay(2000);      // Yellow ON

  // Yellow off, red on
  digitalWrite(light.yellow, LOW);
  digitalWrite(light.red, HIGH);
}
