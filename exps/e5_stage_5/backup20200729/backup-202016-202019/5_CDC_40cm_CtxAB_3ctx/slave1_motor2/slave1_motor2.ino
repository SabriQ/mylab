
#include <Wire.h>

int ena=2;
int dir=3;
int pul=4;
//input
int c_0=5;
int c_1=6;
int c_2=7;

byte num=-1;//default to turn off the led 
//variables
int ctx[3];
int c_ctx;

int de_init = 200;
int de_stop;
int de_stops[3] = {20,25,30};
int de = de_init;
int motor_count_num = 0;
void setup() {
  // put your setup code here, to run once:
pinMode(ena,OUTPUT);digitalWrite(ena,HIGH);
pinMode(dir,OUTPUT);digitalWrite(dir,LOW);
pinMode(pul,OUTPUT);digitalWrite(pul,LOW);
pinMode(c_0,INPUT);
pinMode(c_1,INPUT);
pinMode(c_2,INPUT);

Wire.begin(1);
Wire.onReceive(receiveinfo);
Serial.begin(9600);
}

void loop() {  
//  if (Serial.available()){num = Serial.read();}
//  Serial.println(num);
  rec();
  if (ctx[0]==1){
    c_ctx=0;
  }
  if (ctx[1]==1){
    c_ctx=1;
  }
  if (ctx[2]==1){
    c_ctx=2;
  }
}

void motor_count(){
  motor_count_num = motor_count_num +1;
  if (motor_count_num%3==0){
    de_stop = de_stops[0];
  }else if (motor_count_num%3==1){
    de_stop = de_stops[1];
  }else{
    de_stop = de_stops[2];
  }
}

void rec(){
  switch (num)
  {
    case 0:// go to context 0
      Serial.println("move to context 0");
      digitalWrite(ena,LOW);
      digitalWrite(dir,LOW);//leaving motor
      motor_count();
      do{Read_ctx();pulse_stepper(pul);}while(ctx[0]==0); 
      de = de_init;
      Serial.println(" Done");
      digitalWrite(ena,HIGH);
      c_ctx=0;
      Serial.println(c_ctx);
      num=-1;
      break;
      
    case 1://go to context 1
      Serial.println("move to context 1");
      digitalWrite(ena,LOW);
      if (c_ctx==0){       
        digitalWrite(dir,HIGH);//approaching motor
        }
      if (c_ctx==2){  
        digitalWrite(dir,LOW);//leaving motor        
        }else{;}
       motor_count();
      do{Read_ctx();pulse_stepper(pul);}while(ctx[1]==0);
      de = de_init;
      Serial.println(" Done");
      digitalWrite(ena,HIGH);
      c_ctx=1;
      Serial.println(c_ctx);
      num=-1;
      break;

      case 2://go to context 2
      Serial.println("move to context 2");
      digitalWrite(ena,LOW);
      digitalWrite(dir,HIGH);//approaching motor
      motor_count();
      do{Read_ctx();pulse_stepper(pul);}while(ctx[2]==0); 
      de = de_init;
      Serial.println(" Done");
      digitalWrite(ena,HIGH);
      c_ctx=2;
      Serial.println(c_ctx);
      num=-1;
      break;

    default:
      Read_ctx();
//      if (ctx[0]==1){
//        c_ctx=0;
//      }
//      if (ctx[1]==1){
//        c_ctx=1;
//      }
//      if (ctx[2]==1){
//        c_ctx=2;
//      }
//      if (ctx[0]==0 && ctx[0]==0 && ctx[0]==0){
//        do{Read_ctx();pulse_stepper(pul);}while(ctx[2]==0);
//      }
      break;}}

void Read_ctx(){
  if (Read_digital(c_0,5)>0.9){ctx[0]=1;}else{ctx[0]=0;}
  if (Read_digital(c_1,5)>0.9){ctx[1]=1;}else{ctx[1]=0;}
  if (Read_digital(c_2,5)>0.9){ctx[2]=1;}else{ctx[2]=0;}
//  Serial.print(Read_digital(c_0,10));Serial.print(" ");
//  Serial.print(Read_digital(c_1,10));Serial.print(" ");
//  Serial.println(Read_digital(c_2,10));
//  delay(100);
}
void receiveinfo() {  
  while (Wire.available()) {
    num = Wire.read();
//    if(num==0){go_ctx=0;}
//    else if (num==1){go_ctx=1;}
//    else if (num==2){go_ctx=2;}
//    else{go_ctx = go_ctx;}
    
    Serial.print("Recieve: ");
    Serial.println(num);
    }}

void pulse_stepper2(int port_out)
{
  digitalWrite(port_out, HIGH);
  int one_pulse_time = 500000/400/40;
  
  delayMicroseconds(one_pulse_time );
  //?????????2,???????????????400???pulse;????????????500000us/400;?????????1s2???????????????????????????500000us/400/2
  //delayMicroseconds(483);
  digitalWrite(port_out, LOW);
  delayMicroseconds(one_pulse_time );
  //delayMicroseconds(483);
}


void pulse_stepper(int port_out)
{
  if (de>de_stop){
    de = de-1;
  }
  digitalWrite(port_out, HIGH);
  delayMicroseconds(de);
  digitalWrite(port_out, LOW);
  delayMicroseconds(de);
}

float Read_digital(int digital, int times) {
  float sum = 0;
  for (int i = 0; i < times; i++) {
    int value = digitalRead(digital);
    sum = sum + value;
  }
  return sum / times;
}
