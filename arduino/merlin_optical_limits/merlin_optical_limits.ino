#include <Wire.h>

// accelerometer address
#define MMA8452_ADDRESS 0x1D  // 0x1D if SA0 is high, 0x1C if low

//Define a few of the registers that we will be accessing on the MMA8452
#define OUT_X_MSB 0x01
#define XYZ_DATA_CFG  0x0E
#define WHO_AM_I   0x0D
#define CTRL_REG1  0x2A

// Sets full-scale range to +/-2, 4, or 8g. 
// Used to calc real g values.
#define GSCALE 2 


// ----- PIN definition -----
const int pin0 = A0;   // optical limit switch 0
const int pin1 = A2;   // optical limit switch 1
const int pin2 = A1;   // optical limit switch 2
const int pin3 = A3;   // optical limit switch 3


// Setup executed once 
char dim_label[] = "accelerometer";
void setup()
{ 
  Serial.begin(9600);
  Serial.println("merlin_arduino started");
  
  Wire.begin(); //Join the bus as a master

  initMMA8452(); //Test and intialize the MMA8452
}



void loop()
{
  // ----- Acceleromter -----------
  int accelCount[3];  // Stores the 12-bit signed value
  readAccelData(accelCount);  // Read the x/y/z adc values
  
  // Now we'll calculate the accleration value into actual g's
  float accelG[3];  // Stores the real accel value in g's
  for (int i = 0 ; i < 3 ; i++)
  {
    // get actual g value, this depends on scale being set
    accelG[i] = (float) accelCount[i] / ((1<<12)/(2*GSCALE));  
  }

  // publish x 
  for (int i = 0; i < 3; i++)
  {
    Serial.print(accelG[i],4);  // Print g value
    Serial.print("\t");  // tabs in between 
  }
  Serial.println();

  // ----- Optical Limit Switch -----------
  int state = 0;
  boolean optState[4];
  
  // joint 0
  state = analogRead(pin0);
  if(state < 512){
    optState[0] = true;
  }else{
    optState[0] = false;
  }
  
  // joint 1
  state = analogRead(pin1);
  if(state < 512){
    optState[1] = true;
  }else{
    optState[1] = false;
  }
  
  // joint 2
  state = analogRead(pin2);
  if(state < 512){
    optState[2] = true;
  }else{
    optState[2] = false;
  }
  
  // joint 3
  state = analogRead(pin3);
  if(state < 512){
    optState[3] = true;
  }else{
    optState[3] = false;
  }

//  pub_opt.publish(&msg_opt);
  for (int i = 0; i < 3; i++)
  {
    Serial.print(optState[i]);  // Print g value
    Serial.print("\t");  // tabs in between 
  }
  Serial.println();


  
  delay(10); 
}


void readAccelData(int *destination)
{
  byte rawData[6];  // x/y/z accel register data stored here

  readRegisters(OUT_X_MSB, 6, rawData);  // Read the six raw data registers into data array

  // Loop to calculate 12-bit ADC and g value for each axis
  for(int i = 0; i < 3 ; i++)
  {
    int gCount = (rawData[i*2] << 8) | rawData[(i*2)+1];  //Combine the two 8 bit registers into one 12-bit number
    gCount >>= 4; //The registers are left align, here we right align the 12-bit integer

    // If the number is negative, we have to make it so manually (no 12-bit data type)
    if (rawData[i*2] > 0x7F)
    {  
      gCount = ~gCount + 1;
      gCount *= -1;  // Transform into negative 2's complement #
    }

    destination[i] = gCount; //Record this gCount into the 3 int array
  }
}


// Initialize the MMA8452 registers 
// See the many application notes for more info on setting all of these registers:
void initMMA8452()
{
  byte c = readRegister(WHO_AM_I);  // Read WHO_AM_I register
  if (c == 0x2A) // WHO_AM_I should always be 0x2A
  {  
    Serial.println("MMA8452Q is online...");
  }
  else
  {
    Serial.print("Could not connect to MMA8452Q: 0x");
    Serial.println(c, HEX);
    while(1) ; // Loop forever if communication doesn't happen
  }

  MMA8452Standby();  // Must be in standby to change registers

  // Set up the full scale range to 2, 4, or 8g.
  byte fsr = GSCALE;
  if(fsr > 8) fsr = 8; //Easy error check
  fsr >>= 2; // Neat trick, see page 22. 00 = 2G, 01 = 4A, 10 = 8G
  writeRegister(XYZ_DATA_CFG, fsr);

  //The default data rate is 800Hz and we don't modify it in this example code

  MMA8452Active();  // Set to active to start reading
}

// Sets the MMA8452 to standby mode. It must be in standby to change most register settings
void MMA8452Standby()
{
  byte c = readRegister(CTRL_REG1);
  writeRegister(CTRL_REG1, c & ~(0x01)); //Clear the active bit to go into standby
}

// Sets the MMA8452 to active mode. Needs to be in this mode to output data
void MMA8452Active()
{
  byte c = readRegister(CTRL_REG1);
  writeRegister(CTRL_REG1, c | 0x01); //Set the active bit to begin detection
}

// Read bytesToRead sequentially, starting at addressToRead into the dest byte array
void readRegisters(byte addressToRead, int bytesToRead, byte * dest)
{
  Wire.beginTransmission(MMA8452_ADDRESS);
  Wire.write(addressToRead);
  Wire.endTransmission(false); //endTransmission but keep the connection active

  Wire.requestFrom(MMA8452_ADDRESS, bytesToRead); //Ask for bytes, once done, bus is released by default

  while(Wire.available() < bytesToRead); //Hang out until we get the # of bytes we expect

  for(int x = 0 ; x < bytesToRead ; x++)
    dest[x] = Wire.read();    
}

// Read a single byte from addressToRead and return it as a byte
byte readRegister(byte addressToRead)
{
  Wire.beginTransmission(MMA8452_ADDRESS);
  Wire.write(addressToRead);
  Wire.endTransmission(false); //endTransmission but keep the connection active

  Wire.requestFrom(MMA8452_ADDRESS, 1); //Ask for 1 byte, once done, bus is released by default

  while(!Wire.available()) ; //Wait for the data to come back
  return Wire.read(); //Return this one byte
}

// Writes a single byte (dataToWrite) into addressToWrite
void writeRegister(byte addressToWrite, byte dataToWrite)
{
  Wire.beginTransmission(MMA8452_ADDRESS);
  Wire.write(addressToWrite);
  Wire.write(dataToWrite);
  Wire.endTransmission(); //Stop transmitting
}



