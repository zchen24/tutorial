#include <Wire.h>

#define VCNL4000_ADDRESS 0x13  //I2C Address of the board

void setup(){
  Serial.begin(9600);  // Serial's used to debug and print data
  Wire.begin();  // initialize I2C stuff
  initVCNL4000(); //initilize and setup the board
}

void loop(){
  unsigned int ambientValue = readAmbient(); //can a tiny bit slow
  unsigned int proximityValue = readProximity();

  Serial.print(ambientValue);
  Serial.print(" | ");
  Serial.println(proximityValue);

  delay(100);  //Just here to slow down the printing
  //note that the readings take about 100ms to execute
}






void initVCNL4000(){
  byte temp = readVCNLByte(0x81);

  if (temp != 0x11){  // Product ID Should be 0x11
    Serial.print("initVCNL4000 failed to initialize");
    Serial.println(temp, HEX);
  }else{
    Serial.println("VNCL4000 Online...");
  } 

  /*VNCL400 init params
   Feel free to play with any of these values, but check the datasheet first!*/
  writeVCNLByte(0x84, 0x0F);  // Configures ambient light measures - Single conversion mode, 128 averages
  writeVCNLByte(0x83, 20);  // sets IR current in steps of 10mA 0-200mA --> 200mA
  writeVCNLByte(0x89, 2);  // Proximity IR test signal freq, 0-3 - 781.25 kHz
  writeVCNLByte(0x8A, 0x81);  // proximity modulator timing - 129, recommended by Vishay 
}


unsigned int readProximity(){
  // readProximity() returns a 16-bit value from the VCNL4000's proximity data registers
  byte temp = readVCNLByte(0x80);
  writeVCNLByte(0x80, temp | 0x08);  // command the sensor to perform a proximity measure

  while(!(readVCNLByte(0x80)&0x20));  // Wait for the proximity data ready bit to be set
  unsigned int data = readVCNLByte(0x87) << 8;
  data |= readVCNLByte(0x88);

  return data;
}


unsigned int readAmbient(){
  // readAmbient() returns a 16-bit value from the VCNL4000's ambient light data registers
  byte temp = readVCNLByte(0x80);
  writeVCNLByte(0x80, temp | 0x10);  // command the sensor to perform ambient measure

  while(!(readVCNLByte(0x80)&0x40));  // wait for the proximity data ready bit to be set
  unsigned int data = readVCNLByte(0x85) << 8;
  data |= readVCNLByte(0x86);

  return data;
}


void writeVCNLByte(byte address, byte data){
  // writeVCNLByte(address, data) writes a single byte of data to address
  Wire.beginTransmission(VCNL4000_ADDRESS);
  Wire.write(address);
  Wire.write(data);
  Wire.endTransmission();
}


byte readVCNLByte(byte address){
  // readByte(address) reads a single byte of data from address
  Wire.beginTransmission(VCNL4000_ADDRESS);
  Wire.write(address);
  Wire.endTransmission();
  Wire.requestFrom(VCNL4000_ADDRESS, 1);
  while(!Wire.available());
  byte data = Wire.read();

  return data;
}
