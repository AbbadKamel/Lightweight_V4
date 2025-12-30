// -------------------------------------------------------------------------------
//                          NEAC NMEA 2000 CAN BUS INTERFACE
// Utilisation avec VsCode : https://www.aranacorp.com/fr/programmer-arduino-avec-visual-studio-code/#:~:text=Configuration%20de%20VsCode%20pour%20Arduino&text=Recherchez%20Arduino%2C%20vous%20avez%20alors,plusieurs%20commandes%20relatives%20%C3%A0%20Arduino.&text=En%20bas%20%C3%A0%20droite%2C%20cliquez,Arduino%20(ici%2C%20COM5).
// -------------------------------------------------------------------------------
#define prg_version "Neac - NMEA 2000 Can Bus Reader - V1.0.0"
#define prg_date    "24/08/2024"
#define NEAC_SHIP   1

#include <Arduino.h>

#define ESP32_CAN_TX_PIN GPIO_NUM_5  
#define ESP32_CAN_RX_PIN GPIO_NUM_4 

#include <NMEA2000_CAN.h>
#include <N2kMsg.h>
#include <NMEA2000.h>
#include <sstream>
#include <iomanip>
#include <N2kMessages.h>
#include <N2kMessagesEnumToStr.h>
#include <vector>
#include <unordered_map>
#include <functional>
#include <map>

typedef struct {
  unsigned long PGN;
  void (*Handler)(const tN2kMsg &N2kMsg); 
} tNMEA2000Handler;

// #define DEBUG_MODE

#define READ_STREAM      Serial  
#define FORWARD_STREAM   Serial     // Define ForwardStream to port, what you listen on PC side. On Arduino Due you can use e.g. SerialUSB

Stream *ReadStream    = &READ_STREAM;
Stream *ForwardStream = &FORWARD_STREAM;

#define PI       3.1415926535897932384626433832795
#define MS_TO_KT 1.9438444924406047516198704103672  // Used to convert meters/sec to Knots

unsigned long message_id       = 0;
unsigned long nmea_received_id = 0;
int precision = 6;

int scX = 15;
int dst = 35;
int rf25 = 28;
int cv7 = 32;

#include <ActisenseReader.h>
tActisenseReader ActisenseReader;

void SystemTime                 (const tN2kMsg &N2kMsg);
void Rudder                     (const tN2kMsg &N2kMsg);
void EngineRapid                (const tN2kMsg &N2kMsg);
void EngineDynamicParameters    (const tN2kMsg &N2kMsg);
void TransmissionParameters     (const tN2kMsg &N2kMsg);
void TripFuelConsumption        (const tN2kMsg &N2kMsg);
void Speed                      (const tN2kMsg &N2kMsg);
void WaterDepth                 (const tN2kMsg &N2kMsg);
void BinaryStatus               (const tN2kMsg &N2kMsg);
void FluidLevel                 (const tN2kMsg &N2kMsg);
void OutsideEnvironmental       (const tN2kMsg &N2kMsg);
void Temperature                (const tN2kMsg &N2kMsg);
void TemperatureExt             (const tN2kMsg &N2kMsg);
void DCStatus                   (const tN2kMsg &N2kMsg);
void BatteryConfigurationStatus (const tN2kMsg &N2kMsg);
void COGSOG                     (const tN2kMsg &N2kMsg);
void GNSS                       (const tN2kMsg &N2kMsg);
void LocalOffset                (const tN2kMsg &N2kMsg);
void Attitude                   (const tN2kMsg &N2kMsg);
void Heading                    (const tN2kMsg &N2kMsg);
void Humidity                   (const tN2kMsg &N2kMsg);
void Pressure                   (const tN2kMsg &N2kMsg);
void UserDatumSettings          (const tN2kMsg &N2kMsg);
void GNSSSatsInView             (const tN2kMsg &N2kMsg);

void NavigationInfo             (const tN2kMsg &N2KMsg);
void RouteWPInfo                (const tN2kMsg &N2KMsg);
void CrossTrackError            (const tN2kMsg &N2kMsg);
void RateOfTurn                 (const tN2kMsg &N2kMsg);
void PositionRapid              (const tN2kMsg &N2kMsg);
void WindSpeed                  (const tN2kMsg &N2kMsg);
void GNSSDOPData                (const tN2kMsg &N2kMsg);
void MagneticVariation          (const tN2kMsg &N2kMsg);
void AISClassBPosition          (const tN2kMsg &N2kMsg);

void HeadingTrackControl        (const tN2kMsg &N2kMsg);
void Heave                      (const tN2kMsg &N2kMsg);
void BatteryStatus              (const tN2kMsg &N2kMsg);
void AISstaticdataA             (const tN2kMsg &N2kMsg);
void DirectionData              (const tN2kMsg &N2kMsg);
void VesselSpeedComponents      (const tN2kMsg &N2kMsg);
void AISClassAPosition          (const tN2kMsg &N2kMsg);

std::stringstream stream;
std::string num_str;
tNMEA2000Handler NMEA2000Handlers[]={
  {126992L,&SystemTime},
  {127245L,&Rudder},
  {127250L,&Heading},
  {127251L,&RateOfTurn},
  {127257L,&Attitude},
  {127488L,&EngineRapid},
  {127489L,&EngineDynamicParameters},
  {127493L,&TransmissionParameters},
  {127497L,&TripFuelConsumption},
  {127501L,&BinaryStatus},
  {127505L,&FluidLevel},
  {127506L,&DCStatus},
  {127513L,&BatteryConfigurationStatus},
  {128259L,&Speed},
  {128267L,&WaterDepth},
  {129025L,&PositionRapid},
  {129026L,&COGSOG},
  {129029L,&GNSS},
  {129033L,&LocalOffset},
  {129045L,&UserDatumSettings},
  {129283L,&CrossTrackError},
  {129284L,&NavigationInfo},
  {129285L,&RouteWPInfo},
  {129540L,&GNSSSatsInView},
  {130306L,&WindSpeed},
  {130310L,&OutsideEnvironmental},
  {130312L,&Temperature},
  {130313L,&Humidity},
  {130314L,&Pressure},
  {130316L,&TemperatureExt},
  {129539L,&GNSSDOPData},
  {127258L,&MagneticVariation},

  {127237L,&HeadingTrackControl},
  {127252L,&Heave},
  {127508L,&BatteryStatus},
  {129794L,&AISstaticdataA},
  {130577L,&DirectionData},
  {130578L,&VesselSpeedComponents}, 
  {129798L,&AISClassAPosition},
};

// -------------------------------------------------------------------------------
// TODO : Ne fonctionne pas avec une précision au delà de 4 digit
void printDouble(Stream *ReadStream, double val, unsigned int precision){
  if (int(val)==0 && val<0) ReadStream->print("-"); // Les valeurs entre 0 et -1 perdent leur signe dans la bataille
  ReadStream->print (int(val));
  ReadStream->print(".");
  unsigned int frac;
  if(val >= 0)
      frac = (val - int(val)) * precision;
  else
      frac = (int(val)- val ) * precision;
  ReadStream->print(frac,DEC);
}

/// -------------------------------------------------------------------------------
// TODO : Ne fonctionne pas avec une précision au delà de 4 digit
void printDoubleV2(Stream *ReadStream, double val, unsigned int precision){
  if((abs(val)<pow(10,8)) && (abs(val)>pow(10,-6))){
    
    if (int(val)==0 && val<0) ReadStream->print("-"); // Les valeurs entre 0 et -1 perdent leur signe dans la bataille
    ReadStream->print (int(val));
    ReadStream->print(".");
    unsigned int frac;


    if(val >= 0)
        frac = round((val - int(val)) * pow(10,precision));
    else
        frac = round((int(val)- val ) * pow(10,precision));

    unsigned int decimal = log10(frac)+1;
    unsigned int manquant = precision-decimal;
    if(frac!=0){
      for(int i = 0 ; i<manquant;i++){
        ReadStream->print(0);
      }
    }
    ReadStream->print(frac,DEC);
  }else{
    ReadStream->print("0.0");
  }
}

// -------------------------------------------------------------------------------
// TODO : Ne fonctionne pas avec une précision au delà de 4 digit
double convertDoubleV2(double val, unsigned int precision){
  std::string num_str;
  if((abs(val)<pow(10,8)) && (abs(val)>pow(10,-6))){
    
    if (int(val)==0 && val<0) num_str="-"; // Les valeurs entre 0 et -1 perdent leur signe dans la bataille
      num_str+=std::to_string(int(val));
      num_str+=".";
    unsigned int frac;


    if(val >= 0)
        frac = round((val - int(val)) * pow(10,precision));
    else
        frac = round((int(val)- val ) * pow(10,precision));

    unsigned int decimal = log10(frac)+1;
    unsigned int manquant = precision-decimal;
    if(frac!=0){
      for(int i = 0 ; i<manquant;i++){
        num_str+="0";
      }
      num_str+=std::to_string(frac);
    }
  }else{
    num_str="0.0";
  }
  
  return std::stod(num_str);
}


// -------------------------------------------------------------------------------
void sendMessageToNeacBox(int source,
                          int pgn_id, 
                          char* label_msg, 
                          double value1=0, 
                          double value2=0, 
                          double value3=0, 
                          double value4=0, 
                          double value5=0, 
                          double value6=0, 
                          double value7=0, 
                          double value8=0, 
                          double value9=0){
  // String message_nmea = String(source) + ";" + String(pgn_id) + ";" + (label_msg) + ";" + String(value1);

  // OutputStream->print(F("    -> sendMessageToNeacBox : @"));
  // OutputStream->print(NEAC_SHIP); OutputStream->print(";");
  // OutputStream->print(label_msg); OutputStream->print(";");
  // printDouble(&Serial, value1, 10000);OutputStream->print(";");
  // printDouble(&Serial, value2, 10000);OutputStream->print(";");
  // printDouble(&Serial, value3, 10000);OutputStream->print(";");
  // printDouble(&Serial, value4, 10000);OutputStream->print(";");
  // printDouble(&Serial, message_id, 10000);OutputStream->println("#");
  std::string data = "";
  //data = "a : "+2+ "; b : "+3;
  data="@"+std::to_string(1)+";"+std::to_string(source)+";"+std::to_string(pgn_id)+";"+label_msg+";"+std::to_string(convertDoubleV2(value1,precision))+
  ";"+std::to_string(convertDoubleV2(value2,precision))+";"+std::to_string(convertDoubleV2(value3,precision))+";"+
  std::to_string(convertDoubleV2(value4,precision))+";" ;
  // Serial.print("@");
  // Serial.print(NEAC_SHIP);Serial.print(";");
  // Serial.print(String(source));Serial.print(";");
  // Serial.print(String(pgn_id));Serial.print(";");
  // Serial.print(label_msg);Serial.print(";");

  //********Temporaire , ajouter paramètre dans la fonction************
  // if(pgn_id==129025){
  //   precision = 10000000;
  // }else{
  //   precision = 10000;
  // }
  // printDoubleV2(&Serial, value1, precision);Serial.print(";");
  // printDoubleV2(&Serial, value2, precision);Serial.print(";");
  // printDoubleV2(&Serial, value3, precision);Serial.print(";");
  // printDoubleV2(&Serial, value4, precision);Serial.print(";");
  if(pgn_id==127237){
    data += std::to_string(convertDoubleV2(value5,precision))+";"+std::to_string(convertDoubleV2(value6,precision))+
    ";"+std::to_string(convertDoubleV2(value7,precision))+";";
    // printDoubleV2(&Serial, value5, precision);Serial.print(";");
    // printDoubleV2(&Serial, value6, precision);Serial.print(";");
    // printDoubleV2(&Serial, value7, precision);Serial.print(";");
  }
  data+=std::to_string(message_id)+"#";
  Serial.println(data.c_str());
  // Serial.print(message_id);Serial.println("#");

  message_id++;
}


// --------------------------------------------------------------------------------------
// NMEA 2000
// --------------------------------------------------------------------------------------

template<typename T> void PrintLabelValWithConversionCheckUnDef(const char* label, T val, double (*ConvFunc)(double val)=0, bool AddLf=false, int8_t Desim=-1 ) {
  ReadStream->print(label);
  if (!N2kIsNA(val)) {
    if ( Desim<0 ) {
      if (ConvFunc) { ReadStream->print(ConvFunc(val)); } else { ReadStream->print(val); }
    } else {
      if (ConvFunc) { ReadStream->print(ConvFunc(val),Desim); } else { ReadStream->print(val,Desim); }
    }
  } else ReadStream->print(F("not available"));
  if (AddLf) ReadStream->println();
}

//*****************************************************************************
void SystemTime(const tN2kMsg &N2kMsg) {
    unsigned char SID;
    uint16_t SystemDate;
    double SystemTime;
    tN2kTimeSource TimeSource;
  
    
    if (ParseN2kSystemTime(N2kMsg,SID,SystemDate,SystemTime,TimeSource) ) {
      if(N2kMsg.Source==scX){
        ReadStream->println(F("System Time"));
      
#ifdef DEBUG_MODE      
                      ReadStream->println(F("System Time"));
      PrintLabelValWithConversionCheckUnDef("    SID                    : ",SID,0,true);
      PrintLabelValWithConversionCheckUnDef("    Days since 1.1.1970    : ",SystemDate,0,true);
      PrintLabelValWithConversionCheckUnDef("    Seconds since midnight : ",SystemTime,0,true);
                        ReadStream->print("    Time source            : "); PrintN2kEnumType(TimeSource,ReadStream);
#endif
        SystemTime=convertDoubleV2(SystemTime,3);
        sendMessageToNeacBox(SID, 126992, "SystemTime", double(SystemDate), SystemTime);
      }
    } else {
      ReadStream->print("Failed to parse PGN: "); ReadStream->println(N2kMsg.PGN);
    }
}

//*****************************************************************************

void Rudder(const tN2kMsg &N2kMsg) {
    unsigned char            instance;
    tN2kRudderDirectionOrder rudder_direction_order;
    double                   rudder_position;
    double                   angle_order;
  
  
    if (ParseN2kRudder(N2kMsg,rudder_position, instance, rudder_direction_order, angle_order) ) {
      if(N2kMsg.Source==rf25){
#ifdef DEBUG_MODE
      PrintLabelValWithConversionCheckUnDef("Rudder - Instance ", instance, 0, true);
      PrintLabelValWithConversionCheckUnDef("    Position (deg)    : ", rudder_position, &RadToDeg, true);
                        ReadStream->print("    Direction order   : "); PrintN2kEnumType(rudder_direction_order, ReadStream);
      PrintLabelValWithConversionCheckUnDef("    Angle order (deg) : ", angle_order, &RadToDeg, true);
#endif
        rudder_position=convertDoubleV2(rudder_position,4);
        if (rudder_position > -1000000000){
          sendMessageToNeacBox(instance, 127245, "Rudder", rudder_position, angle_order);
        }
      }
    } else {
#ifdef DEBUG_MODE      
      ReadStream->print("Failed to parse PGN: "); ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************

void EngineRapid(const tN2kMsg &N2kMsg) {
    unsigned char EngineInstance;
    double EngineSpeed;
    double EngineBoostPressure;
    int8_t EngineTiltTrim;
  
    
    if (ParseN2kEngineParamRapid(N2kMsg,EngineInstance, EngineSpeed, EngineBoostPressure, EngineTiltTrim) ) {
#ifdef DEBUG_MODE      
      PrintLabelValWithConversionCheckUnDef("Engine rapid params - Instance", EngineInstance, 0, true);
      PrintLabelValWithConversionCheckUnDef("    RPM                 : ", EngineSpeed,0,true);
      PrintLabelValWithConversionCheckUnDef("    Boost pressure (Pa) : ",EngineBoostPressure,0,true);
      PrintLabelValWithConversionCheckUnDef("    Tilt trim           : ",EngineTiltTrim,0,true);
#endif      
        sendMessageToNeacBox(EngineInstance, 127488, "EngineRapid" , EngineSpeed, EngineBoostPressure, double(EngineTiltTrim));
    } else {
#ifdef DEBUG_MODE      
      ReadStream->print("Failed to parse PGN: "); ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void EngineDynamicParameters(const tN2kMsg &N2kMsg) {
    unsigned char EngineInstance;
    double EngineOilPress;
    double EngineOilTemp;
    double EngineCoolantTemp;
    double AltenatorVoltage;
    double FuelRate;
    double EngineHours;
    double EngineCoolantPress;
    double EngineFuelPress; 
    int8_t EngineLoad;
    int8_t EngineTorque;
    tN2kEngineDiscreteStatus1 Status1;
    tN2kEngineDiscreteStatus2 Status2;
  
  
    if (ParseN2kEngineDynamicParam(N2kMsg,EngineInstance,EngineOilPress,EngineOilTemp,EngineCoolantTemp,
                                 AltenatorVoltage,FuelRate,EngineHours,
                                 EngineCoolantPress,EngineFuelPress,
                                 EngineLoad,EngineTorque,Status1,Status2) ) {
#ifdef DEBUG_MODE                                  
      PrintLabelValWithConversionCheckUnDef("Engine dynamic params: ",EngineInstance,0,true);
      PrintLabelValWithConversionCheckUnDef("  oil pressure (Pa): ",EngineOilPress,0,true);
      PrintLabelValWithConversionCheckUnDef("  oil temp (C): ",EngineOilTemp,&KelvinToC,true);
      PrintLabelValWithConversionCheckUnDef("  coolant temp (C): ",EngineCoolantTemp,&KelvinToC,true);
      PrintLabelValWithConversionCheckUnDef("  altenator voltage (V): ",AltenatorVoltage,0,true);
      PrintLabelValWithConversionCheckUnDef("  fuel rate (l/h): ",FuelRate,0,true);
      PrintLabelValWithConversionCheckUnDef("  engine hours (h): ",EngineHours,&SecondsToh,true);
      PrintLabelValWithConversionCheckUnDef("  coolant pressure (Pa): ",EngineCoolantPress,0,true);
      PrintLabelValWithConversionCheckUnDef("  fuel pressure (Pa): ",EngineFuelPress,0,true);
      PrintLabelValWithConversionCheckUnDef("  engine load (%): ",EngineLoad,0,true);
      PrintLabelValWithConversionCheckUnDef("  engine torque (%): ",EngineTorque,0,true);
#endif      
      //sendMessageToNeacBox("EngineDynamicParams" , EngineOilPress, EngineOilTemp, EngineCoolantTemp, AltenatorVoltage, FuelRate, EngineHours, EngineCoolantPress, EngineFuelPress, EngineLoad, EngineTorque);      
    } else {
#ifdef DEBUG_MODE      
      ReadStream->print("Failed to parse PGN: "); ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void TransmissionParameters(const tN2kMsg &N2kMsg) {
    unsigned char EngineInstance;
    tN2kTransmissionGear TransmissionGear;
    double OilPressure;
    double OilTemperature;
    unsigned char DiscreteStatus1;
  
  
    if (ParseN2kTransmissionParameters(N2kMsg,EngineInstance, TransmissionGear, OilPressure, OilTemperature, DiscreteStatus1) ) {
#ifdef DEBUG_MODE      
      PrintLabelValWithConversionCheckUnDef("Transmission params: ",EngineInstance,0,true);
                        ReadStream->print(F("  gear: ")); PrintN2kEnumType(TransmissionGear,ReadStream);
      PrintLabelValWithConversionCheckUnDef("  oil pressure (Pa): ",OilPressure,0,true);
      PrintLabelValWithConversionCheckUnDef("  oil temperature (C): ",OilTemperature,&KelvinToC,true);
      PrintLabelValWithConversionCheckUnDef("  discrete status: ",DiscreteStatus1,0,true);
#endif      
      //sendMessageToNeacBox("TransmissionParams" , TransmissionGear, OilPressure, OilTemperature, DiscreteStatus1);      
    } else {
#ifdef DEBUG_MODE      
      ReadStream->print(F("Failed to parse PGN: ")); ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void TripFuelConsumption(const tN2kMsg &N2kMsg) {
    unsigned char EngineInstance;
    double TripFuelUsed;
    double FuelRateAverage; 
    double FuelRateEconomy; 
    double InstantaneousFuelEconomy; 
  
  
    if (ParseN2kEngineTripParameters(N2kMsg,EngineInstance, TripFuelUsed, FuelRateAverage, FuelRateEconomy, InstantaneousFuelEconomy) ) {
#ifdef DEBUG_MODE      
      PrintLabelValWithConversionCheckUnDef("Trip fuel consumption: ",EngineInstance,0,true);
      PrintLabelValWithConversionCheckUnDef("  fuel used (l): ",TripFuelUsed,0,true);
      PrintLabelValWithConversionCheckUnDef("  average fuel rate (l/h): ",FuelRateAverage,0,true);
      PrintLabelValWithConversionCheckUnDef("  economy fuel rate (l/h): ",FuelRateEconomy,0,true);
      PrintLabelValWithConversionCheckUnDef("  instantaneous fuel economy (l/h): ",InstantaneousFuelEconomy,0,true);
#endif      
      sendMessageToNeacBox(EngineInstance, 127497, "TripFuelConsumption" , TripFuelUsed, FuelRateAverage, FuelRateEconomy, InstantaneousFuelEconomy);            
    } else {
#ifdef DEBUG_MODE      
      ReadStream->print(F("Failed to parse PGN: ")); ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void Heading(const tN2kMsg &N2kMsg) {
    unsigned char SID;
    tN2kHeadingReference headingReference;
    double               heading=0;
    double               deviation=0;
    double               variation=0;
  
  
    if (ParseN2kHeading(N2kMsg, SID, heading, deviation, variation, headingReference) ) {
#ifdef DEBUG_MODE      
                      ReadStream->println("Heading");
      PrintLabelValWithConversionCheckUnDef("    SID             : ",SID,0,true);
                        ReadStream->print("    Reference       : "); PrintN2kEnumType(headingReference,ReadStream);
      PrintLabelValWithConversionCheckUnDef("    Heading (deg)   : ", heading,   &RadToDeg, true);
      PrintLabelValWithConversionCheckUnDef("    Deviation (deg) : ", deviation, &RadToDeg, true);
      PrintLabelValWithConversionCheckUnDef("    Variation (deg) : ", variation, &RadToDeg, true);
#endif      
      sendMessageToNeacBox(SID, 127250, "Heading" , heading / PI * 180);
    } else {
#ifdef DEBUG_MODE      
      ReadStream->print(F("Failed to parse PGN: ")); ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void COGSOG(const tN2kMsg &N2kMsg) {
    // PGN 129026
    unsigned char SID;
    tN2kHeadingReference HeadingReference;
    double COG;
    double SOG;
  
  
    if (ParseN2kCOGSOGRapid(N2kMsg,SID,HeadingReference,COG,SOG) ) {
#ifdef DEBUG_MODE      
                      ReadStream->println(F("COG/SOG"));
      PrintLabelValWithConversionCheckUnDef("    SID       : ",SID,0,true);
                          ReadStream->print("    Reference : "); PrintN2kEnumType(HeadingReference,ReadStream);
      PrintLabelValWithConversionCheckUnDef("    COG (deg) : ", COG, &RadToDeg, true);
      PrintLabelValWithConversionCheckUnDef("    SOG (m/s) : ", SOG, 0        , true);
#else
      // TODO : COG (deg) : not available alors ne pas envoyer la valeur COG brute
      sendMessageToNeacBox(SID, 129026, "COGSOG", COG/PI*180, SOG);
#endif      
    } else {
#ifdef DEBUG_MODE      
      ReadStream->print(F("Failed to parse PGN: ")); ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void GNSS(const tN2kMsg &N2kMsg) {
    unsigned char SID;
    uint16_t DaysSince1970;
    double SecondsSinceMidnight; 
    double Latitude;
    double Longitude;
    double Altitude; 
    tN2kGNSStype GNSStype;
    tN2kGNSSmethod GNSSmethod;
    unsigned char nSatellites;
    double HDOP;
    double PDOP;
    double GeoidalSeparation;
    unsigned char nReferenceStations;
    tN2kGNSStype ReferenceStationType;
    uint16_t ReferenceSationID;
    double AgeOfCorrection;
  
  
    if (ParseN2kGNSS(N2kMsg,SID,DaysSince1970,SecondsSinceMidnight,
                  Latitude,Longitude,Altitude,
                  GNSStype,GNSSmethod,
                  nSatellites,HDOP,PDOP,GeoidalSeparation,
                  nReferenceStations,ReferenceStationType,ReferenceSationID,
                  AgeOfCorrection) ) {
#ifdef DEBUG_MODE                     
                      ReadStream->println("GNSS info");
      PrintLabelValWithConversionCheckUnDef("    SID                    : ",SID,0,true);
      PrintLabelValWithConversionCheckUnDef("    Days since 1.1.1970    : ",DaysSince1970,0,true);
      PrintLabelValWithConversionCheckUnDef("    Seconds since midnight : ",SecondsSinceMidnight,0,true);
      PrintLabelValWithConversionCheckUnDef("    Latitude           : ",Latitude,0,true,9);
      PrintLabelValWithConversionCheckUnDef("    Longitude          : ",Longitude,0,true,9);
      PrintLabelValWithConversionCheckUnDef("    Altitude           : (m): ",Altitude,0,true);
                        ReadStream->print("    GNSS type          : "); PrintN2kEnumType(GNSStype,ReadStream);
                        ReadStream->print("    GNSS method        : "); PrintN2kEnumType(GNSSmethod,ReadStream);
      PrintLabelValWithConversionCheckUnDef("    Satellite count    : ",nSatellites,0,true);
      PrintLabelValWithConversionCheckUnDef("    HDOP               : ",HDOP,0,true);
      PrintLabelValWithConversionCheckUnDef("    PDOP               : ",PDOP,0,true);
      PrintLabelValWithConversionCheckUnDef("    Geoidal separation : ",GeoidalSeparation,0,true);
      PrintLabelValWithConversionCheckUnDef("    Reference stations : ",nReferenceStations,0,true);
#endif      
      sendMessageToNeacBox(SID, 129029, "GNSSInfo", Latitude, Longitude, SecondsSinceMidnight,Altitude);
    } else {
#ifdef DEBUG_MODE       
      ReadStream->print(F("Failed to parse PGN: ")); ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void UserDatumSettings(const tN2kMsg &N2kMsg) {
  if (N2kMsg.PGN!=129045L) return;
  int Index=0;
  double val;
#ifdef DEBUG_MODE 
  ReadStream->println(F("User Datum Settings: "));
  val=N2kMsg.Get4ByteDouble(1e-2,Index);
  PrintLabelValWithConversionCheckUnDef("  delta x (m): ",val,0,true);
  val=N2kMsg.Get4ByteDouble(1e-2,Index);
  PrintLabelValWithConversionCheckUnDef("  delta y (m): ",val,0,true);
  val=N2kMsg.Get4ByteDouble(1e-2,Index);
  PrintLabelValWithConversionCheckUnDef("  delta z (m): ",val,0,true);
  val=N2kMsg.GetFloat(Index);
  PrintLabelValWithConversionCheckUnDef("  rotation in x (deg): ",val,&RadToDeg,true,5);
  val=N2kMsg.GetFloat(Index);
  PrintLabelValWithConversionCheckUnDef("  rotation in y (deg): ",val,&RadToDeg,true,5);
  val=N2kMsg.GetFloat(Index);
  PrintLabelValWithConversionCheckUnDef("  rotation in z (deg): ",val,&RadToDeg,true,5);
  val=N2kMsg.GetFloat(Index);
  PrintLabelValWithConversionCheckUnDef("  scale: ",val,0,true,3);
#endif
}

//*****************************************************************************
void GNSSSatsInView(const tN2kMsg &N2kMsg) {
  unsigned char SID;
  tN2kRangeResidualMode Mode;
  uint8_t NumberOfSVs;
  tSatelliteInfo SatelliteInfo;

  if (ParseN2kPGNSatellitesInView(N2kMsg,SID,Mode,NumberOfSVs) ) {
#ifdef DEBUG_MODE     
    ReadStream->println("Satellites in view: ");
                      ReadStream->print("  mode: "); ReadStream->println(Mode);
                      ReadStream->print("  number of satellites: ");  ReadStream->println(NumberOfSVs);
    for ( uint8_t i=0; i<NumberOfSVs && ParseN2kPGNSatellitesInView(N2kMsg,i,SatelliteInfo); i++) {
                        ReadStream->print("  Satellite PRN: ");  ReadStream->println(SatelliteInfo.PRN);
      PrintLabelValWithConversionCheckUnDef("    elevation: ",SatelliteInfo.Elevation,&RadToDeg,true,1);
      PrintLabelValWithConversionCheckUnDef("    azimuth:   ",SatelliteInfo.Azimuth,&RadToDeg,true,1);
      PrintLabelValWithConversionCheckUnDef("    SNR:       ",SatelliteInfo.SNR,0,true,1);
      PrintLabelValWithConversionCheckUnDef("    residuals: ",SatelliteInfo.RangeResiduals,0,true,1);
                        ReadStream->print("    status: "); ReadStream->println(SatelliteInfo.UsageStatus);
    }
#endif 
  } 
}

//*****************************************************************************
void LocalOffset(const tN2kMsg &N2kMsg) {
    uint16_t SystemDate;
    double SystemTime;
    int16_t Offset;
  
      if (ParseN2kLocalOffset(N2kMsg,SystemDate,SystemTime,Offset) ) {
#ifdef DEBUG_MODE       
                      ReadStream->println("Date,time and local offset: ");
      PrintLabelValWithConversionCheckUnDef("  days since 1.1.1970: ",SystemDate,0,true);
      PrintLabelValWithConversionCheckUnDef("  seconds since midnight: ",SystemTime,0,true);
      PrintLabelValWithConversionCheckUnDef("  local offset (min): ",Offset,0,true);
#endif            
    } else {
#ifdef DEBUG_MODE       
      ReadStream->print("Failed to parse PGN: "); ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void OutsideEnvironmental(const tN2kMsg &N2kMsg) {
    unsigned char SID;
    double WaterTemperature;
    double OutsideAmbientAirTemperature;
    double AtmosphericPressure;
  
  
    if (ParseN2kOutsideEnvironmentalParameters(N2kMsg,SID,WaterTemperature,OutsideAmbientAirTemperature,AtmosphericPressure) ) {
#ifdef DEBUG_MODE       
      PrintLabelValWithConversionCheckUnDef("Water temp: ",WaterTemperature,&KelvinToC);
      PrintLabelValWithConversionCheckUnDef(", outside ambient temp: ",OutsideAmbientAirTemperature,&KelvinToC);
      PrintLabelValWithConversionCheckUnDef(", pressure: ",AtmosphericPressure,0,true);
#endif
      if(OutsideAmbientAirTemperature<400){
        sendMessageToNeacBox(SID, 130310, "OutsideEnvironmental" , WaterTemperature, OutsideAmbientAirTemperature, AtmosphericPressure);
      }
    } else {
#ifdef DEBUG_MODE 
      ReadStream->print("Failed to parse PGN: ");  ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void Temperature(const tN2kMsg &N2kMsg) {
    unsigned char SID;
    unsigned char TempInstance;
    tN2kTempSource TempSource;
    double ActualTemperature;
    double SetTemperature;
  
  
    if (ParseN2kTemperature(N2kMsg,SID,TempInstance,TempSource,ActualTemperature,SetTemperature) ) {
                        ReadStream->print("Temperature source: "); PrintN2kEnumType(TempSource,ReadStream,false);
#ifdef DEBUG_MODE                         
      PrintLabelValWithConversionCheckUnDef(", actual temperature: ",ActualTemperature,&KelvinToC);
      PrintLabelValWithConversionCheckUnDef(", set temperature: ",SetTemperature,&KelvinToC,true);
#endif      
      sendMessageToNeacBox(SID, 130312, "Temperature", ActualTemperature, NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
    } else {
#ifdef DEBUG_MODE
      ReadStream->print("Failed to parse PGN: ");  ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void Humidity(const tN2kMsg &N2kMsg) {
    unsigned char SID;
    unsigned char Instance;
    tN2kHumiditySource HumiditySource;
    double ActualHumidity,SetHumidity;
  
  
    if ( ParseN2kHumidity(N2kMsg,SID,Instance,HumiditySource,ActualHumidity,SetHumidity) ) {
                        ReadStream->print("Humidity source: "); PrintN2kEnumType(HumiditySource,ReadStream,false);
#ifdef DEBUG_MODE                         
      PrintLabelValWithConversionCheckUnDef(", humidity: ",ActualHumidity,0,false);
      PrintLabelValWithConversionCheckUnDef(", set humidity: ",SetHumidity,0,true);
#endif            
      sendMessageToNeacBox(SID, 130313, "Humidity", ActualHumidity, NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
    } else {
#ifdef DEBUG_MODE       
      ReadStream->print("Failed to parse PGN: ");  ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void Pressure(const tN2kMsg &N2kMsg) {
    unsigned char SID;
    unsigned char Instance;
    tN2kPressureSource PressureSource;
    double ActualPressure;
  
  
    if ( ParseN2kPressure(N2kMsg,SID,Instance,PressureSource,ActualPressure) ) {
                        ReadStream->print("Pressure source: "); PrintN2kEnumType(PressureSource,ReadStream,false);
#ifdef DEBUG_MODE                         
      PrintLabelValWithConversionCheckUnDef(", pressure: ",ActualPressure,&PascalTomBar,true);
#endif
      sendMessageToNeacBox(SID, 130314, "Pressure", ActualPressure, NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
    } else {
#ifdef DEBUG_MODE 
      ReadStream->print("Failed to parse PGN: ");  ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void TemperatureExt(const tN2kMsg &N2kMsg) {
    unsigned char SID;
    unsigned char TempInstance;
    tN2kTempSource TempSource;
    double ActualTemperature;
    double SetTemperature;
  
  
    if (ParseN2kTemperatureExt(N2kMsg,SID,TempInstance,TempSource,ActualTemperature,SetTemperature) ) {
                        ReadStream->print("Temperature source: "); PrintN2kEnumType(TempSource,ReadStream,false);
#ifdef DEBUG_MODE                         
      PrintLabelValWithConversionCheckUnDef(", actual temperature: ",ActualTemperature,&KelvinToC);
      PrintLabelValWithConversionCheckUnDef(", set temperature: ",SetTemperature,&KelvinToC,true);
#endif
      if(ActualTemperature<314 && TempSource==0){
        sendMessageToNeacBox(SID, 130316, "TemperatureExt", ActualTemperature, NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
      }
    } else {
#ifdef DEBUG_MODE      
      ReadStream->print("Failed to parse PGN: ");  ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void BatteryConfigurationStatus(const tN2kMsg &N2kMsg) {
    unsigned char BatInstance;
    tN2kBatType BatType;
    tN2kBatEqSupport SupportsEqual;
    tN2kBatNomVolt BatNominalVoltage;
    tN2kBatChem BatChemistry;
    double BatCapacity;
    int8_t BatTemperatureCoefficient;
    double PeukertExponent; 
    int8_t ChargeEfficiencyFactor;
  
  
    if (ParseN2kBatConf(N2kMsg,BatInstance,BatType,SupportsEqual,BatNominalVoltage,BatChemistry,BatCapacity,BatTemperatureCoefficient,PeukertExponent,ChargeEfficiencyFactor) ) {
#ifdef DEBUG_MODE       
      PrintLabelValWithConversionCheckUnDef("Battery instance: ",BatInstance,0,true);
                        ReadStream->print("  - type: "); PrintN2kEnumType(BatType,ReadStream);
                        ReadStream->print("  - support equal.: "); PrintN2kEnumType(SupportsEqual,ReadStream);
                        ReadStream->print("  - nominal voltage: "); PrintN2kEnumType(BatNominalVoltage,ReadStream);
                        ReadStream->print("  - chemistry: "); PrintN2kEnumType(BatChemistry,ReadStream);
      PrintLabelValWithConversionCheckUnDef("  - capacity (Ah): ",BatCapacity,&CoulombToAh,true);
      PrintLabelValWithConversionCheckUnDef("  - temperature coefficient (%): ",BatTemperatureCoefficient,0,true);
      PrintLabelValWithConversionCheckUnDef("  - peukert exponent: ",PeukertExponent,0,true);
      PrintLabelValWithConversionCheckUnDef("  - charge efficiency factor (%): ",ChargeEfficiencyFactor,0,true);
#endif      
    } else {
#ifdef DEBUG_MODE       
      ReadStream->print("Failed to parse PGN: "); ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void DCStatus(const tN2kMsg &N2kMsg) {
    unsigned char SID;
    unsigned char DCInstance;
    tN2kDCType DCType;
    unsigned char StateOfCharge;
    unsigned char StateOfHealth;
    double TimeRemaining;
    double RippleVoltage;
    double Capacity;
  
  
    if (ParseN2kDCStatus(N2kMsg,SID,DCInstance,DCType,StateOfCharge,StateOfHealth,TimeRemaining,RippleVoltage,Capacity) ) {
#ifdef DEBUG_MODE       
      ReadStream->print("DC instance: ");
      ReadStream->println(DCInstance);
      ReadStream->print("  - type: "); PrintN2kEnumType(DCType,ReadStream);
      ReadStream->print("  - state of charge (%): "); ReadStream->println(StateOfCharge);
      ReadStream->print("  - state of health (%): "); ReadStream->println(StateOfHealth);
      ReadStream->print("  - time remaining (h): "); ReadStream->println(TimeRemaining/60);
      ReadStream->print("  - ripple voltage: "); ReadStream->println(RippleVoltage);
      ReadStream->print("  - capacity: "); ReadStream->println(Capacity);
#endif
    } else {
#ifdef DEBUG_MODE       
      ReadStream->print("Failed to parse PGN: ");  ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void Speed(const tN2kMsg &N2kMsg) {
    unsigned char SID;
    double SOW;
    double SOG;
    tN2kSpeedWaterReferenceType SWRT;
  

    if (ParseN2kBoatSpeed(N2kMsg,SID,SOW,SOG,SWRT) ) {
      if(N2kMsg.Source==dst){
#ifdef DEBUG_MODE       
      ReadStream->print("Boat speed");
      PrintLabelValWithConversionCheckUnDef("    SOW :",N2kIsNA(SOW)?SOW:msToKnots(SOW));
      PrintLabelValWithConversionCheckUnDef("    SOG :",N2kIsNA(SOG)?SOG:msToKnots(SOG));
      ReadStream->print(", ");
      PrintN2kEnumType(SWRT,ReadStream,true);
#endif
      sendMessageToNeacBox(SID, 128259, "Speed" , SOW, SOG);
      }
    }
}

//*****************************************************************************
void WaterDepth(const tN2kMsg &N2kMsg) {
    unsigned char SID;
    double DepthBelowTransducer;
    double Offset;
  
  
    if (ParseN2kWaterDepth(N2kMsg,SID,DepthBelowTransducer,Offset) ) {
      if ( N2kIsNA(Offset) || Offset == 0 ) {
#ifdef DEBUG_MODE         
        PrintLabelValWithConversionCheckUnDef("Depth below transducer",DepthBelowTransducer);
        if ( N2kIsNA(Offset) ) {
          ReadStream->println(", offset not available");
        } else {
          ReadStream->println(", offset=0");
        }
#endif
        sendMessageToNeacBox(SID, 128267, "WaterDepth", DepthBelowTransducer);
      } else {
#ifdef DEBUG_MODE         
        if (Offset>0) {
          ReadStream->print("Water depth:");
        } else {
          ReadStream->print("Depth below keel:");
        }
        if ( !N2kIsNA(DepthBelowTransducer) ) { 
          ReadStream->println(DepthBelowTransducer+Offset); 
        } else {  ReadStream->println(" not available"); }
#endif
      }
    }
}

//*****************************************************************************
void printLLNumber(Stream *ReadStream, unsigned long long n, uint8_t base=10)
{
  unsigned char buf[16 * sizeof(long)]; // Assumes 8-bit chars.
  unsigned long long i = 0;

  if (n == 0) {
    ReadStream->print('0');
    return;
  }

  while (n > 0) {
    buf[i++] = n % base;
    n /= base;
  }

  for (; i > 0; i--)
    ReadStream->print((char) (buf[i - 1] < 10 ?
      '0' + buf[i - 1] :
      'A' + buf[i - 1] - 10));
}

//*****************************************************************************
void BinaryStatusFull(const tN2kMsg &N2kMsg) {
    unsigned char BankInstance;
    tN2kBinaryStatus BankStatus;

    if (ParseN2kBinaryStatus(N2kMsg,BankInstance,BankStatus) ) {
#ifdef DEBUG_MODE       
      ReadStream->print("Binary status for bank "); ReadStream->print(BankInstance); ReadStream->println(":");
      ReadStream->print("  "); //printLLNumber(ReadStream,BankStatus,16);
      for (uint8_t i=1; i<=28; i++) {
        if (i>1) ReadStream->print(",");
        PrintN2kEnumType(N2kGetStatusOnBinaryStatus(BankStatus,i),ReadStream,false);
      }
      ReadStream->println();
#endif
    }
}

//*****************************************************************************
void BinaryStatus(const tN2kMsg &N2kMsg) {
    unsigned char BankInstance;
    tN2kOnOff Status1,Status2,Status3,Status4;
  
  
    if (ParseN2kBinaryStatus(N2kMsg,BankInstance,Status1,Status2,Status3,Status4) ) {
      if (BankInstance>2) { // note that this is only for testing different methods. MessageSender.ini sends 4 status for instace 2
        BinaryStatusFull(N2kMsg);
      } else {
#ifdef DEBUG_MODE         
        ReadStream->print("Binary status for bank "); ReadStream->print(BankInstance); ReadStream->println(":");
        ReadStream->print("  Status1=");PrintN2kEnumType(Status1,ReadStream,false);
        ReadStream->print(", Status2=");PrintN2kEnumType(Status2,ReadStream,false);
        ReadStream->print(", Status3=");PrintN2kEnumType(Status3,ReadStream,false);
        ReadStream->print(", Status4=");PrintN2kEnumType(Status4,ReadStream,false);
        ReadStream->println();
#endif
      }
    }
}

//*****************************************************************************
void FluidLevel(const tN2kMsg &N2kMsg) {
    unsigned char Instance;
    tN2kFluidType FluidType;
    double Level=0;
    double Capacity=0;
  
  
    if (ParseN2kFluidLevel(N2kMsg,Instance,FluidType,Level,Capacity) ) {
#ifdef DEBUG_MODE       
      switch (FluidType) {
        case N2kft_Fuel:
          ReadStream->print("Fuel level :");
          break;
        case N2kft_Water:
          ReadStream->print("Water level :");
          break;
        case N2kft_GrayWater:
          ReadStream->print("Gray water level :");
          break;
        case N2kft_LiveWell:
          ReadStream->print("Live well level :");
          break;
        case N2kft_Oil:
          ReadStream->print("Oil level :");
          break;
        case N2kft_BlackWater:
          ReadStream->print("Black water level :");
          break;
        case N2kft_FuelGasoline:
          ReadStream->print("Gasoline level :");
          break;
        case N2kft_Error:
          ReadStream->print("Error level :");
          break;
        case N2kft_Unavailable:
          ReadStream->print("Unknown level :");
          break;
      }
      ReadStream->print(Level); ReadStream->print("%"); 
      ReadStream->print(" ("); ReadStream->print(Capacity*Level/100); ReadStream->print("l)");
      ReadStream->print(" capacity :"); ReadStream->println(Capacity);
#endif
    }
}

//*****************************************************************************
void Attitude(const tN2kMsg &N2kMsg) {
    unsigned char SID;
    double Yaw;
    double Pitch;
    double Roll;
  
  
    if (ParseN2kAttitude(N2kMsg, SID, Yaw, Pitch, Roll) ) {
#ifdef DEBUG_MODE       
                        ReadStream->println("Attitude");
      PrintLabelValWithConversionCheckUnDef("    SID         : ",SID,0,true);
      PrintLabelValWithConversionCheckUnDef("    Yaw (deg)   : ",Yaw,&RadToDeg,true);
      PrintLabelValWithConversionCheckUnDef("    Pitch (deg) : ",Pitch,&RadToDeg,true);
      PrintLabelValWithConversionCheckUnDef("    Roll (deg)  : ",Roll,&RadToDeg,true);
#endif
      sendMessageToNeacBox(SID, 127257, "Attitude" ,Yaw, Pitch, Roll);
    } else {
#ifdef DEBUG_MODE
      ReadStream->print("Failed to parse PGN: "); ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void NavigationInfo(const tN2kMsg &N2kMsg) {
    unsigned char               SID;
    double                      DistanceToWaypoint;
    tN2kHeadingReference        BearingReference;
    bool                        PerpendicularCrossed;
    bool                        ArrivalCircleEntered;
    tN2kDistanceCalculationType CalculationType;
    double                      ETATime;
    int16_t                     ETADate;
    double                      BearingOriginToDestinationWaypoint;
    double                      BearingPositionToDestinationWaypoint;
    uint32_t                    OriginWaypointNumber;
    uint32_t                     DestinationWaypointNumber;
    double                      DestinationLatitude;
    double                      DestinationLongitude;
    double                      WaypointClosingVelocity;
   
  
  
    if (ParseN2kNavigationInfo(N2kMsg, SID, DistanceToWaypoint, BearingReference,
                               PerpendicularCrossed, ArrivalCircleEntered, CalculationType,
                               ETATime, ETADate, BearingOriginToDestinationWaypoint, BearingPositionToDestinationWaypoint,
                               OriginWaypointNumber,  DestinationWaypointNumber,
                               DestinationLatitude, DestinationLongitude,  WaypointClosingVelocity) ) {
#ifdef DEBUG_MODE                                 
      ReadStream->println("Navigation Info"); 
      PrintLabelValWithConversionCheckUnDef("    DistanceToWaypoint                   : ", DistanceToWaypoint, 0, true);
      PrintLabelValWithConversionCheckUnDef("    BearingReference                     : ", BearingReference, 0, true);
      PrintLabelValWithConversionCheckUnDef("    PerpendicularCrossed                 : ", PerpendicularCrossed, 0, true);
      PrintLabelValWithConversionCheckUnDef("    ArrivalCircleEntered                 : ", ArrivalCircleEntered, 0, true);
      //ReadStream->print                  ("    CalculationType                      : "); PrintN2kEnumType(CalculationType,ReadStream, false);ReadStream->println("");
      PrintLabelValWithConversionCheckUnDef("    ETATime                              : ", ETATime, 0, true);
      PrintLabelValWithConversionCheckUnDef("    ETADate                              : ", ETADate, 0, true);
      PrintLabelValWithConversionCheckUnDef("    BearingOriginToDestWaypoint (Deg)    : ", BearingOriginToDestinationWaypoint, &RadToDeg, true);
      PrintLabelValWithConversionCheckUnDef("    BearingPositionToDestWaypoint (Deg)  : ", BearingPositionToDestinationWaypoint, &RadToDeg, true);
      PrintLabelValWithConversionCheckUnDef("    OriginWaypointNumber                 : ", OriginWaypointNumber, 0, true);
      PrintLabelValWithConversionCheckUnDef("    DestinationWaypointNumber            : ", DestinationWaypointNumber, 0, true);
      PrintLabelValWithConversionCheckUnDef("    DestinationLatitude                  : ", DestinationLatitude, 0, true);
      PrintLabelValWithConversionCheckUnDef("    DestinationLongitude                 : ", DestinationLongitude, 0, true);
      PrintLabelValWithConversionCheckUnDef("    WaypointClosingVelocity              : ", WaypointClosingVelocity, 0, true);
#endif
      //sendMessageToNeacBox("NavigationInfo" ,DistanceToWaypoint,BearingReference,PerpendicularCrossed,ArrivalCircleEntered,BearingOriginToDestinationWaypoint,BearingPositionToDestinationWaypoint,DestinationLatitude,DestinationLongitude,WaypointClosingVelocity);
    } else {
#ifdef DEBUG_MODE       
      ReadStream->print(F("  Failed to parse PGN: "));  ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void RouteWPInfo(const tN2kMsg &N2kMsg) {
    unsigned char               SID;
    double                      DistanceToWaypoint;
    tN2kHeadingReference        BearingReference;
    bool                        PerpendicularCrossed;
    bool                        ArrivalCircleEntered;
    tN2kDistanceCalculationType CalculationType;
    double                      ETATime;
    int16_t                     ETADate;
    double                      BearingOriginToDestinationWaypoint;
    double                      BearingPositionToDestinationWaypoint;
    uint32_t                     OriginWaypointNumber;
    uint32_t                     DestinationWaypointNumber;
    double                      DestinationLatitude;
    double                      DestinationLongitude;
    double                      WaypointClosingVelocity;
   
  
  
    if (ParseN2kNavigationInfo(N2kMsg, SID, DistanceToWaypoint, BearingReference,
                               PerpendicularCrossed, ArrivalCircleEntered, CalculationType,
                               ETATime, ETADate, BearingOriginToDestinationWaypoint, BearingPositionToDestinationWaypoint,
                               OriginWaypointNumber,  DestinationWaypointNumber,
                               DestinationLatitude, DestinationLongitude,  WaypointClosingVelocity) ) {
#ifdef DEBUG_MODE                                 
      ReadStream->println("RouteWPInfo"); 
      PrintLabelValWithConversionCheckUnDef("    DistanceToWaypoint                   : ", DistanceToWaypoint, 0, true);
      PrintLabelValWithConversionCheckUnDef("    BearingReference                     : ", BearingReference, 0, true);
      PrintLabelValWithConversionCheckUnDef("    PerpendicularCrossed                 : ", PerpendicularCrossed, 0, true);
      PrintLabelValWithConversionCheckUnDef("    ArrivalCircleEntered                 : ", ArrivalCircleEntered, 0, true);
     // ReadStream->print("  CalculationType: "); PrintN2kEnumType(CalculationType,ReadStream);
      //PrintLabelValWithConversionCheckUnDef("  CalculationType : ", CalculationType, &tN2kDistanceCalculationType, true);
      PrintLabelValWithConversionCheckUnDef("    ETATime                              : ", ETATime, 0, true);
      PrintLabelValWithConversionCheckUnDef("    ETADate                              : ", ETADate, 0, true);
      PrintLabelValWithConversionCheckUnDef("    BearingOriginToDestinationWaypoint   : ", BearingOriginToDestinationWaypoint, 0, true);
      PrintLabelValWithConversionCheckUnDef("    BearingPositionToDestinationWaypoint : ", BearingPositionToDestinationWaypoint, 0, true);
      PrintLabelValWithConversionCheckUnDef("    OriginWaypointNumber                 : ", OriginWaypointNumber, 0, true);
      PrintLabelValWithConversionCheckUnDef("    DestinationWaypointNumber            : ", DestinationWaypointNumber, 0, true);
      PrintLabelValWithConversionCheckUnDef("    DestinationLatitude                  : ", DestinationLatitude, 0, true);
      PrintLabelValWithConversionCheckUnDef("    DestinationLongitude                 : ", DestinationLongitude, 0, true);
      PrintLabelValWithConversionCheckUnDef("    WaypointClosingVelocity              : ", WaypointClosingVelocity, 0, true);
#endif
      //sendMessageToNeacBox("NavigationInfo" ,DistanceToWaypoint,BearingReference,PerpendicularCrossed,ArrivalCircleEntered,BearingOriginToDestinationWaypoint,BearingPositionToDestinationWaypoint,DestinationLatitude,DestinationLongitude,WaypointClosingVelocity);
    } else {
#ifdef DEBUG_MODE       
      ReadStream->print("  Failed to parse PGN: ");  ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void CrossTrackError(const tN2kMsg &N2kMsg) {
    unsigned char   SID;
    tN2kXTEMode     XTEMode;
    bool            NavigationTerminated;
    double          XTE_value;
  
  
    if (ParseN2kXTE(N2kMsg, SID, XTEMode, NavigationTerminated, XTE_value) ) {
#ifdef DEBUG_MODE       
      ReadStream->println(F("Cross Trak Error"));
      //PrintLabelValWithConversionCheckUnDef("    SID                  : ", SID, 0, true);
      //ReadStream->print                  ("    XTEMode              : "); PrintN2kEnumType(XTEMode,ReadStream, false);ReadStream->println(""); 
      //PrintLabelValWithConversionCheckUnDef("    NavigationTerminated : ", NavigationTerminated, 0, true);
      PrintLabelValWithConversionCheckUnDef("    XTE_value                  : ", XTE_value, 0, true);
#endif      
      //sendMessageToNeacBox("CrossTrackError", NavigationTerminated, XTE_value, XTEMode);*/
    } else {
#ifdef DEBUG_MODE       
      ReadStream->print(F("  Failed to parse PGN: "));  ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void RateOfTurn(const tN2kMsg &N2kMsg) {
    unsigned char   SID;
    double          RateOfTurn;
  
  
    if (ParseN2kRateOfTurn(N2kMsg, SID, RateOfTurn) ) {
#ifdef DEBUG_MODE       
      ReadStream->println("Rate Of Turn"); 
      PrintLabelValWithConversionCheckUnDef("    RateOfTurn : ", RateOfTurn, 0, true);
#endif
      sendMessageToNeacBox(SID, 127251, "RateOfTurn" ,RateOfTurn);
    } else {
#ifdef DEBUG_MODE
      ReadStream->print("  Failed to parse PGN: ");  ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void PositionRapid(const tN2kMsg &N2kMsg) {
    double latitude;
    double longitude;
    if (ParseN2kPositionRapid(N2kMsg, latitude, longitude) ) {
#ifdef DEBUG_MODE       
      ReadStream->println("Position Rapid"); 
      PrintLabelValWithConversionCheckUnDef("    Latitude  : ", latitude,  0, true);
      PrintLabelValWithConversionCheckUnDef("    Longitude : ", longitude, 0, true);
#endif
      sendMessageToNeacBox(0, 129025, "PositionRapid", latitude, longitude);
    } else {
#ifdef DEBUG_MODE       
      ReadStream->print("  Failed to parse PGN: ");  ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void WindSpeed(const tN2kMsg &N2kMsg) {
    unsigned char     SID;
    double            WindSpeed;
    double            WindAngle;
    tN2kWindReference WindReference;
  
  
    if (ParseN2kWindSpeed(N2kMsg, SID, WindSpeed, WindAngle, WindReference) ) {
#ifdef DEBUG_MODE       
      ReadStream->println("Wind Speed"); 
      PrintLabelValWithConversionCheckUnDef("    WindSpeed (Kn)  : ", WindSpeed, &msToKnots, true, 2);
      PrintLabelValWithConversionCheckUnDef("    WindAngle (deg) : ", WindAngle, &RadToDeg,  true, 0);
      //PrintLabelValWithConversionCheckUnDef("    WindReference   : ", WindReference, 0, true);
      //                  ReadStream->print("    WindReference   : "); PrintN2kEnumType(WindReference,ReadStream);
#endif      
      //sendMessageToNeacBox(SID, 130306, "WindSpeed", WindSpeed * MS_TO_KT, WindAngle / PI * 180);
      WindSpeed=convertDoubleV2(WindSpeed,2);
      WindAngle=convertDoubleV2(WindAngle,4);

      sendMessageToNeacBox(SID, 130306, "WindSpeed", WindSpeed, WindAngle);

    } else {
#ifdef DEBUG_MODE       
      ReadStream->print("  Failed to parse PGN: ");  ReadStream->println(N2kMsg.PGN);
#endif
    }
}

//*****************************************************************************
void GNSSDOPData(const tN2kMsg &N2kMsg) {
  //ReadStream->print("GNSS DOP Data");
  //ReadStream->println("GNSS DOP Data");

    unsigned char    SID;/*
    tN2kGNSSDOPmode  DesiredMode;
    tN2kGNSSDOPmode  ActualMode;
    double           HDOP;
    double           VDOP;
    double           TDOP;*/

    //if (ParseN2kGNSSDOPData(N2kMsg, SID, DesiredMode, ActualMode, HDOP, VDOP, TDOP)) {
      //ReadStream->println("GNSS DOP Data");
      //ReadStream->print                  ("    DesiredMode : "); PrintN2kEnumType(DesiredMode,ReadStream, false);ReadStream->println(""); 
      //ReadStream->print                  ("    ActualMode  : "); PrintN2kEnumType(ActualMode,ReadStream, false);ReadStream->println(""); 
      //PrintLabelValWithConversionCheckUnDef("    HDOP        : ", HDOP, 0, true);
      //PrintLabelValWithConversionCheckUnDef("    VDOP        : ", VDOP, 0, true);
      //PrintLabelValWithConversionCheckUnDef("    TDOP        : ", TDOP, 0, true);
    //} else {
    //  ReadStream->print("  Failed to parse PGN: ");  ReadStream->println(N2kMsg.PGN);
    //}
}

//*****************************************************************************
void MagneticVariation(const tN2kMsg &N2kMsg) {
  unsigned char         SID;
  tN2kMagneticVariation Source;
  uint16_t              DaysSince1970;
  double                Variation;

  
  if (ParseN2kMagneticVariation(N2kMsg, SID, Source, DaysSince1970, Variation)){
#ifdef DEBUG_MODE     
    ReadStream->println("MagneticVariation"); 
    ReadStream->print                  ("    Source        : "); PrintN2kEnumType(Source,ReadStream, false);
    PrintLabelValWithConversionCheckUnDef("    Variation     : ", Variation, 0, true);
    //PrintLabelValWithConversionCheckUnDef("    DaysSince1970 : ", DaysSince1970, 0, true);
#endif
    sendMessageToNeacBox(SID, 127258, "MagneticVariation", Variation);
  }
}

//*****************************************************************************
void AISClassBPosition(const tN2kMsg &N2kMsg) {
  uint8_t       MessageID;
  tN2kAISRepeat Repeat;
  uint32_t      UserID;
  double        Latitude;
  double        Longitude;
  bool          Accuracy;
  bool          RAIM;
  uint8_t       Seconds;
  double        COG;
  double        SOG;
  tN2kAISTransceiverInformation AISTransceiverInformation;
  double        Heading;
  tN2kAISUnit   Unit;
  bool          Display;
  bool          DSC;
  bool          Band;
  bool          Msg22;
  tN2kAISMode   Mode;
  bool          State;
  /*
  if (ParseN2kAISClassBPosition(N2kMsg, MessageID, Repeat, UserID, Latitude, Longitude, Accuracy, RAIM, Seconds, COG, 
                                SOG, AISTransceiverInformation, Heading, Unit, Display, DSC, Band, Msg22, Mode, State)){
    ReadStream->println("AISClassBPosition"); 
    PrintLabelValWithConversionCheckUnDef("    MessageID     : ", MessageID, 0, true);
    PrintLabelValWithConversionCheckUnDef("    Latitude      : ", Latitude, 0, true);
    PrintLabelValWithConversionCheckUnDef("    Longitude     : ", Longitude, 0, true);
    PrintLabelValWithConversionCheckUnDef("    Accuracy      : ", Accuracy, 0, true);
    PrintLabelValWithConversionCheckUnDef("    COG           : ", COG, 0, true);
    PrintLabelValWithConversionCheckUnDef("    SOG           : ", SOG, 0, true);
    //PrintLabelValWithConversionCheckUnDef("    DaysSince1970 : ", DaysSince1970, 0, true);
    sendMessageToNeacBox("AISClassBPosition", Latitude, Longitude, COG, SOG);
  }*/
}

//*****************************************************************************
void HeadingTrackControl(const tN2kMsg &N2kMsg) {
  unsigned char         SID;
  tN2kOnOff RudderLimitExceeded;
  tN2kOnOff OffHeadingLimitExceeded;
  tN2kOnOff OffTrackLimitExceeded;
  tN2kOnOff Override;
  tN2kSteeringMode SteeringMode;
  tN2kTurnMode TurnMode;
  tN2kHeadingReference HeadingReference;
  tN2kRudderDirectionOrder CommandedRudderDirection;
  double CommandedRudderAngle;
  double HeadingToSteerCourse;
  double Track;
  double RudderLimit;
  double OffHeadingLimit;
  double RadiusOfTurnOrder;
  double RateOfTurnOrder;
  double OffTrackLimit;
  double VesselHeading;

  
  if (ParseN2kPGN127237(N2kMsg, RudderLimitExceeded,
                        OffHeadingLimitExceeded,
                        OffTrackLimitExceeded,
                        Override,
                        SteeringMode,
                        TurnMode,
                        HeadingReference,
                        CommandedRudderDirection,
                        CommandedRudderAngle,
                        HeadingToSteerCourse,
                        Track,
                        RudderLimit,
                        OffHeadingLimit,
                        RadiusOfTurnOrder,
                        RateOfTurnOrder,
                        OffTrackLimit,
                        VesselHeading)){
#ifdef DEBUG_MODE     
    ReadStream->println("HeadingTrackControl"); 
    //ReadStream->print                  ("    Source        : "); PrintN2kEnumType(Source,ReadStream, false);
    // TODO : décodege à faire, mais pas de valeur pour le moment sur le bateau, donc pas utile
    //printLabelValWithConversionCheckUnDef("    Variation     : ", Variation, 0, true);
    //PrintLabelValWithConversionCheckUnDef("    DaysSince1970 : ", DaysSince1970, 0, true);
#endif
    sendMessageToNeacBox(SID, 127237, "HeadingTrackControl", RateOfTurnOrder,SteeringMode,TurnMode,CommandedRudderDirection,CommandedRudderAngle,
    RadiusOfTurnOrder,HeadingToSteerCourse);
  }
}

//*****************************************************************************
void Heave(const tN2kMsg &N2kMsg) {
  unsigned char         SID;
  double                Heave;


  if (ParseN2kHeave(N2kMsg, SID, Heave)){
#ifdef DEBUG_MODE     
    ReadStream->println("Heave"); 
    PrintLabelValWithConversionCheckUnDef("    Heave     : ", Heave, 0, true);
#endif
    sendMessageToNeacBox(SID, 127252, "Heave", Heave);
  }
}

//*****************************************************************************
void BatteryStatus(const tN2kMsg &N2kMsg) {
  unsigned char         SID;
  unsigned char         BatteryInstance;
  double                BatteryVoltage;
  double                BatteryCurrent;
  double                BatteryTemperature;

  
  if (ParseN2kPGN127508(N2kMsg, BatteryInstance, BatteryVoltage, BatteryCurrent, BatteryTemperature, SID)){
#ifdef DEBUG_MODE     
    ReadStream->println("BatteryStatus"); 
    PrintLabelValWithConversionCheckUnDef("    BatteryInstance        : ", BatteryInstance, 0, true);
    PrintLabelValWithConversionCheckUnDef("    BatteryVoltage         : ", BatteryVoltage, 0, true);
    PrintLabelValWithConversionCheckUnDef("    BatteryCurrent         : ", BatteryCurrent, 0, true);
    PrintLabelValWithConversionCheckUnDef("    BatteryTemperature     : ", BatteryTemperature, 0, true);
#endif
    sendMessageToNeacBox(SID, 127508, "BatteryStatus", BatteryInstance, BatteryVoltage, BatteryCurrent, BatteryTemperature);
  }
}

//*****************************************************************************
void AISstaticdataA(const tN2kMsg &N2kMsg) {
  uint8_t MessageID;
  tN2kAISRepeat Repeat;
  uint32_t UserID;
  uint32_t IMOnumber;
  char* Callsign;
  size_t CallsignBufSize;
  char* Name;
  size_t NameBufSize;
  uint8_t VesselType;
  double Length;
  double Beam; 
  double PosRefStbd; 
  double PosRefBow; 
  uint16_t ETAdate; 
  double ETAtime;
  double Draught; 
  char *Destination; 
  size_t DestinationBufSize;
  tN2kAISVersion AISversion; 
  tN2kGNSStype GNSStype;
  tN2kAISDTE DTE; 
  tN2kAISTransceiverInformation AISinfo;
  uint8_t SID;

  if (ParseN2kPGN129794(N2kMsg, MessageID, Repeat, UserID,
                        IMOnumber, Callsign,CallsignBufSize, Name,NameBufSize, VesselType, Length,
                        Beam, PosRefStbd, PosRefBow, ETAdate, ETAtime,
                        Draught, Destination,DestinationBufSize, AISversion, GNSStype,
                        DTE, AISinfo,SID)){
#ifdef DEBUG_MODE     
    ReadStream->println("AISstaticdataA"); 
    // PrintLabelValWithConversionCheckUnDef("    Heave     : ", Heave, 0, true);
    // TODO : Décodege à faire, mais pas de valeur pour le moment sur le bateau, donc pas utile
#endif
    sendMessageToNeacBox(SID, 129794, "AISstaticdataA", 0);
  }
}

//*****************************************************************************
void DirectionData(const tN2kMsg &N2kMsg) {
  unsigned char         SID;
  tN2kDataMode          DataMode;
  tN2kHeadingReference  CogReference;
  double                COG;
  double                SOG;
  double                Heading;
  double                SpeedThroughWater;
  double                Set;
  double                Drift;
  
  if (ParseN2kPGN130577(N2kMsg, DataMode, CogReference, SID, COG, SOG, Heading, SpeedThroughWater, Set, Drift)){
#ifdef DEBUG_MODE     
    ReadStream->println("DirectionData"); 
    PrintLabelValWithConversionCheckUnDef("    COG                   : ", COG, 0, true);
    PrintLabelValWithConversionCheckUnDef("    SOG                   : ", SOG, 0, true);
    PrintLabelValWithConversionCheckUnDef("    Heading               : ", Heading, 0, true);
    PrintLabelValWithConversionCheckUnDef("    SpeedThroughWater     : ", SpeedThroughWater, 0, true);
    PrintLabelValWithConversionCheckUnDef("    Set                   : ", Set, 0, true);
    PrintLabelValWithConversionCheckUnDef("    Drift                 : ", Drift, 0, true);
#endif
    sendMessageToNeacBox(SID, 130577, "DirectionData", COG, SOG, Heading, SpeedThroughWater, Set, Drift);
  }
}

//*****************************************************************************
void VesselSpeedComponents(const tN2kMsg &N2kMsg) {
  unsigned char         SID;
  // Todo : PGN non implémenté dans la librairie NMEA 2000
  sendMessageToNeacBox(0, 0, "VesselSpeedComponents", 0);
}

//*****************************************************************************
void AISClassAPosition(const tN2kMsg &N2kMsg) {
  unsigned char         SID;
  // Todo : PGN non implémenté dans la librairie NMEA 2000
  sendMessageToNeacBox(0, 0, "AISClassAPosition", 0);
}

// ----------------------------------------------------------------
void HandleStreamN2kMsg(const tN2kMsg &N2kMsg) {
  int iHandler;
#ifdef DEBUG_MODE  
  ForwardStream->println("_______________________________");
  ForwardStream->print(F("PGN (#")); 
  printDouble(&Serial, nmea_received_id++, 10000);
  ForwardStream->print(F(") ID = ")); 
  ForwardStream->print(N2kMsg.PGN); ForwardStream->print(F(" - "));
#endif

  for (iHandler=0; NMEA2000Handlers[iHandler].PGN!=0 && !(N2kMsg.PGN==NMEA2000Handlers[iHandler].PGN); iHandler++);
  if (NMEA2000Handlers[iHandler].PGN!=0) {
    NMEA2000Handlers[iHandler].Handler(N2kMsg); 
  } else {
#ifdef DEBUG_MODE
    ForwardStream->println(F("Unknown PGN")); 
#endif
  }
}
/*
std::unordered_map<std::string, std::function<void()>> allSendMethod;
void sendToPgn(std::string pgn){
    tN2kMsg N2kMsg;
    N2kMsg.Priority=2;  // Définir la priorité
    N2kMsg.Source=99;  // Définir la source
    N2kMsg.Destination=255;  // Définir la destination
    //SetN2kLatLonRapid(N2kMsg, 48.8583, 2.2945);
    if(pgn=="129025"){
        SetN2kLatLonRapid(N2kMsg, 48.8583, 2.2945);
    }else if(pgn=="127245"){
        //SetN2kRudder(N2kMsg, 0, 0, 0);
    }
    NMEA2000.SendMsg(N2kMsg);

}
std::map<std::string,std::vector<double> > datatoSendMap;

void sendToPgn(std::string pgn,std::vector<double> dataToSend){
  
    if(pgn=="129025"){
        //SetN2kLatLonRapid(N2kMsg, 48.8583, 2.2945);
    }else if(pgn=="127245"){
        //SetN2kRudder(N2kMsg, 0, 0, 0);
    }
    //NMEA2000.SendMsg(N2kMsg);

}
// ----------------------------------------------------------------
void msgToSend(){
    std::string data ="";
    while(Serial.available() > 0){
        data += Serial.readString().c_str();
        // sendMessageToNeacBox(0, 0, "Test", 0);
    }

    int numData = 0;
    std::string pgn = 0;
    int instance = 0;
    std::vector<double> dataToSend;
    std::string str = "Hello;World;C++";
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(data);
    while (std::getline(tokenStream, data, ';')) {
        tokens.push_back(token);
    }   
    if(data != "" && data.find("$") != std::string::npos && data.find("#") != std::string::npos){
        while(tokens.size() > numData){
            if(tokens[numData] == "$"){ 
            }else if(tokens[numData] == "#"){
                datatoSendMap[pgn] = dataToSend;
                sendToPgn(pgn, dataToSend);
                numData = 0;
                pgn = "";
                instance = 0;
                dataToSend.clear();
            }else if(numData == 1){
                pgn = tokens[numData];
            }else if(numData == 2){
                instance = atoi(tokens[numData].c_str());
            }else{
                dataToSend.push_back(atof(tokens[numData].c_str()));
            }
            numData++;
        }
        // Serial.println(data);
        // sendMessageToNeacBox(0, 0, "Test", 0);
    }
    // Définir les détails du message
  
  // SetN2kPGN129025(N2kMsg}

}
*/

// *****************************************************************************
void SendN2kMsg(const tN2kMsg &N2kMsg) {
  if ( NMEA2000.SendMsg(N2kMsg) ) {
    N2kMsgSentCount++;
  } else {
    N2kMsgFailCount++;
  }
  //if ( ShowSentMessages ) N2kMsg.Print(&Serial);
}

// *****************************************************************************
// Function check is it time to send message. If it is, message will be sent and
// next send time will be updated.
// Function always returns next time it should be handled.
int64_t CheckSendMessage(tN2kSendMessage &N2kSendMessage) {
  if ( N2kSendMessage.Scheduler.IsDisabled() ) return N2kSendMessage.Scheduler.GetNextTime();

  if ( N2kSendMessage.Scheduler.IsTime() ) {
    N2kSendMessage.Scheduler.UpdateNextTime();
    N2kSendMessage.SendFunction();
  }

  return N2kSendMessage.Scheduler.GetNextTime();
}

// *****************************************************************************
// Function send enabled messages from tN2kSendMessage structure according to their
// period+offset.
void SendN2kMessages() {
  static uint64_t NextSend=0;
  uint64_t Now=N2kMillis64();

  if ( NextSend<Now ) {
    uint64_t NextMsgSend;
    NextSend=Now+2000;
    for ( size_t i=0; i<nN2kSendMessages; i++ ) {
      NextMsgSend=CheckSendMessage(N2kSendMessages[i]);
      if ( NextMsgSend<NextSend ) NextSend=NextMsgSend;
    }
  }
}


std::map<std::string, std::vector<std::string>> mapPgn;
// ----------------------------------------------------------------
void msgToSend(){
    std::string data ="";
    while(Serial.available() > 0){
        data += Serial.readString().c_str();
    }    
    std::vector<std::vector<std::string>> lines;
    std::string token;
    std::istringstream tokenStream(data);
    // while (std::getline(tokenStream, token, ';')) {
    //     tokens.push_back(token);
    //     Serial.println(token.c_str()); 
    // }
    while (std::getline(tokenStream, token)) {
        std::istringstream lineStream(token);
        std::vector<std::string> line;
        Serial.println(token.c_str());
        while (std::getline(lineStream, token,';')) {
          line.push_back(token);
        }
        lines.push_back(line);
    }
    for (const auto &it : lines) { 
      if(it[0] == "$" && it[it.size()-1] == "#"){
        mapPgn[it[1]] = it;
        //Serial.println(it[1].c_str());
      }
    }
    
}


// ----------------------------------------------------------------
void setup() {
#ifdef DEBUG_MODE 
  Serial.begin(115200);   // Serial Port for debugging
  Serial.println("Neac - NMEA 2000 Can Bus Reader");
  Serial.println(prg_version);
  Serial.println(prg_date);
  delay(1000);
#endif
  delay(2000);
  NMEA2000.SetN2kCANSendFrameBufSize(250);
  // Set Product information
  NMEA2000.SetProductInformation("00000001", // Manufacturer's Model serial code
                                 100, // Manufacturer's product code
                                 "Message sender example",  // Manufacturer's Model ID
                                 "1.1.2.35 (2022-10-01)",  // Manufacturer's Software version code
                                 "1.1.2.0 (2022-10-01)" // Manufacturer's Model version
                                 );
  // Set device information
  NMEA2000.SetDeviceInformation(1, // Unique number. Use e.g. Serial number.
                                132, // Device function=Analog to NMEA 2000 Gateway. See codes on https://web.archive.org/web/20190531120557/https://www.nmea.org/Assets/20120726%20nmea%202000%20class%20&%20function%20codes%20v%202.00.pdf
                                25, // Device class=Inter/Intranetwork Device. See codes on  https://web.archive.org/web/20190531120557/https://www.nmea.org/Assets/20120726%20nmea%202000%20class%20&%20function%20codes%20v%202.00.pdf
                                2046 // Just choosen free from code list on https://web.archive.org/web/20190529161431/http://www.nmea.org/Assets/20121020%20nmea%202000%20registration%20list.pdf
                               );
  //NMEA2000.SetN2kCANSendFrameBufSize(150);
  //NMEA2000.SetN2kCANReceiveFrameBufSize(150);
  if (ReadStream!=ForwardStream) READ_STREAM.begin(115200);
  FORWARD_STREAM.begin(115200);
  NMEA2000.SetForwardType(tNMEA2000::fwdt_Text); // dpi
  NMEA2000.SetForwardStream(ForwardStream); 
  NMEA2000.SetMsgHandler(HandleStreamN2kMsg);
  NMEA2000.SetMode(tNMEA2000::N2km_ListenAndSend);

  if (ReadStream==ForwardStream) NMEA2000.SetForwardOwnMessages(false); // If streams are same, do not echo own messages.
  NMEA2000.EnableForward(false);
  NMEA2000.Open();


  // Serial.println("NMEA 2000 Can Bus Initialized");
}

// ----------------------------------------------------------------
void loop() {
  if ( Sending ) SendN2kMessages();
  NMEA2000.ParseMessages();  
  CheckCommand();
  CheckLoopTime();
  //sendToPgn("129025");
}
