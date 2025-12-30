/*
Neac-Rudder-Controller
Propriété NEAC-INDUSTRY
Codé par François Chanu et Nicolas Kerthe

input:
PGN 127245 – Rudder 
Field 1: Rudder Instance – This field is used to identify the rudder instance number 
and ranges between 0 and 251. 
2: Direction Order – This field identifies a directional command contained in this 
message. The RAA100 ships from the factory with a default value of 0x0 indicating 
that no direction order is contained in this message. 
3: Reserved – This field is reserved by NMEA 
4: Angle Order – This field is used to indicate an angle order directed towards a 
rudder actuator The RAA100 ships from the factory with a default value of 0x7FFF 
indicating that no angle order is present in this message. 
5: Position – This field is used to indicate the current angle of the rudder in units of 
0.0001 radians. 
6: Reserved – This field is reserved by NMEA

1: instance de le cas d'un second appareil émetteur, dans notre cas ont peut mettre   au moins 10 pour éviter les conflits 
2: stand-by ou consigne a prendre en compte 
4: notre consigne d'angle envoyé par l'aspedsa ou 0x7fff si retour au centre par exemple 
5: notre fameux feedback d'angle en retour de la recopie Rf25

> field 2 : mode de commande de l'Aspedsa :
0 à 2: Standby
3: Neac Rudder Order
5: Neac Cap Order

Mode:
0 à 2: Standby
3: Neac Rudder Order
4: Manual Rudder
5: Neac Cap Order
6: Manual Cap
7: Set angle local

PGN 130577 Message "Direction Data"
 * The purpose of this PGN is to group three fundamental vectors related
 * to vessel motion, speed and heading referenced to the water, speed and
 * course referenced to ground and current speed and flow direction.

output:
DC motor by IBT driver (BTS 7960)

*/
// ===============================================================================
// DEBUG
// Décommenter la ligne ci-dessous pour avoir les logs concernant la télécommande :
// ===============================================================================
#define DEBUG_RC_BARRE
#define DEBUG_RC_GAZ
#define DEBUG_RC_PROPULSEUR

const int ESCAPE = 32;
int bouton4 = ESCAPE;

const int VALID = 33;
int bouton2 = VALID;  //bouton de validation du comptage

const int DOWN = 27;
int bouton3 = DOWN;  //décrémentation du comptage

const int UP = 13;
int bouton1 = UP;  //incrémentation du comptage

const int LEDGREEN = 23;
//#define LEDRED 33
//#define LEDYELL 33
const int LEDWHIT = 25;
//#define LEDBLUE 39
const int RPWM = 4;  // define pin 3 for RPWM pin (output)
const int R_EN = 0;  // define pin 2 for R_EN pin (input)
//#define R_IS 8       // define pin 5 for R_IS pin (output)
const int LPWM = 2;   // define pin 6 for LPWM pin (output)
const int L_EN = 15;  // define pin 7 for L_EN pin (input)
//#define L_IS 11      // define pin 8 for L_IS pin (output)
const int STARBORD = 26;  // relais BABORD
const int PORTSIDE = 28;  // relais TRIBORD
//#define JB 32       // Joystick bouton
//#define JX 34       // Joystick axe X
//#define JY 35       // Joystick axe Y

double angle_order_neac;  // en Radian envoyé par l'Aspedsa
//double AngleOrder = 0;        // en Radian
double Angle_Order = 0;       // en Degrès
double rudder_position_nmea;  // en Radian depuis Pgn
double RudderPosition = 0;    // en Radian
double Rudder_Position;       // en Degrès
//double throttle_neac = 0;     //consigne de gaz depuis Aspedsa -100 à +100%
//double thruster_neac = 0;     //consigne propulseur étrave depuis Aspedsa -100 à +100%
int Instance = 10;
int RudderDirectionOrder = 0;
int rudder_direction_order;
int Mode = 0;  // 0 à 2: Standby, 3: Neac Rudder Order, 4: Manual Rudder, 5: Neac Cap Order, 6: Manual Cap, 7: Set angle local, ...
int Set_Angle = 0;
unsigned char SID;
int Heading = 100;
int Cap_Order = 120;
int Ecart = 0;
int Speed = 45;
int RudFeedback = false;
// https://ihm3d.fr/httpwww-ihm3d-frle-bouton-poussoir.html
int comptage = 0;
int setA = 0;
int ETATBP1;  //variable etat bouton
int ETATBP2;  //variable etat bouton
int ETATBP3;  //variable etat bouton
int ETATBP4;  //variable etat bouton
int JoyB;
int JoyX;
int JoyY;
bool IsOpen = false;

//Aspedsa Remote:
int rudder_order = -1;
double rudder_value;
int throttle_order = -1;
double throttle;
int thruster;

#include <Arduino.h>

//Library for the OLED Display
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ---- DAC
#include <MCP4725.h>  // MCP4725 by Rob Tillaart <rob.tillaart@gmail.com> Version 0.3.7 INSTALLED
#include <string.h>
#include <Wire.h>

//Servo
//#include <Servo.h>
#include "sbus.h"

/* SBUS object, reading SBUS */
// SbusRx(HardwareSerial *bus, const int8_t rxpin, const int8_t txpin, const bool inv)
bfs::SbusRx sbus_rx(&Serial2, 16, 17, true);
/* SBUS object, writing SBUS */
//bfs::SbusTx sbus_tx(&Serial2);
/* SBUS data */
bfs::SbusData data;

#define adresseI2cDacGaz 0x60   // Adresse i2c du module MCP4725
MCP4725 dac(adresseI2cDacGaz);  // Avec adresse I2C du MCP4725 = adresse i2c de votre carte MCP 4725

#define adresseI2cDacThruster 0x61    // Adresse i2c du module MCP4725 thruster
MCP4725 dac2(adresseI2cDacThruster);  // Avec adresse I2C du DACthruster = adresse i2c de votre carte MCP 4725 thruster

//Servo servo_thruster;  // Déclare la sortie servo

int tiller_value = 0;       // Voie 1 = Commande de barre
int gaz_value = 1500;       // Voie 2 = Manette des gaz
int mooring_value = 1500;   // Voie 2 = Manette des gaz
int thruster_value = 1500;  // Voie 3 = Propulseur d'étrave
int rear;
int modeRC;
int killman;
int tiller_offset;
int channel8;

double nmea_rudder_value = 0.00;
int8_t EngineTilt = 0.00;
int8_t ThrusterTilt = 0.00;

int valeurEnvoyeeAuDacGaz;       // Valeur 12 bits (0..4095) permettant de balayer les 4096 niveaux de tension possible du DAC
int valeurEnvoyeeAuDacThruster;  //modif FC

int consigne_dac_gaz = 1500;       // Gaz à 0%
int consigne_dac_thruster = 1500;  //modif FC

int thruster_joystick_value;  // valeur joystick thruster sur le pupitre
int servomanuel;              // consigne thruster manuel
int servoradio;               // consigne thruster télecommandé
int is_DAC_init = 0;          // 1 if DAC is successfully initialized
int is_DAC2_init = 0;         // 1 if DAC is successfully initialized

#define OLED_RESET 4
Adafruit_SSD1306 display(OLED_RESET);

// Version 1.3, 04.08.2020, AK-Homberger

//#define ESP32_CAN_TX_PIN GPIO_NUM_19;  // Set CAN TX port to 5 (Caution!!! Pin 2 before)
//#define ESP32_CAN_RX_PIN GPIO_NUM_18;  // Set CAN RX port to 4
//int RX_PIN = 18;
//int TX_PIN = 19;
#define gpio_matrix_in(RX_PIN,CAN_RX_IDX);
#define gpio_matrix_out(TX_PIN,CAN_TX_IDX);
#define ESP32_CAN_TX_PIN
#define ESP32_CAN_TX_PIN GPIO_NUM_19
#define ESP32_CAN_RX_PIN
#define ESP32_CAN_RX_PIN GPIO_NUM_18

#include <Arduino.h>
#include <NMEA2000_CAN.h>  // This will automatically choose right CAN library and create suitable NMEA2000 object
#include <N2kMessages.h>   // Set the information for other bus devices, which messages we support

typedef struct {
  unsigned long PGN;
  void (*Handler)(const tN2kMsg &N2kMsg);
} tNMEA2000Handler;

void Rudder(const tN2kMsg &N2kMsg);
void Attitude(const tN2kMsg &N2kMsg);

tNMEA2000Handler NMEA2000Handlers[] = {
  { 127257L, &Attitude },
  { 127245L, &Rudder }
};

//-----------------------------------------------------------------------------------------------------
void Rudder(const tN2kMsg &N2kMsg) {
  unsigned char instance;
  tN2kRudderDirectionOrder rudder_direction_order;
  //double rudder_position_nmea;
  //double angle_order_neac;
  unsigned char instancev2 = 10;
  unsigned char source;
  Serial.println("Nikolaiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii");

  if (ParseN2kRudder(N2kMsg, rudder_position_nmea, instance, rudder_direction_order, angle_order_neac)) {
    if (instance == instancev2) {  /////////********//////
      Instance = instance;
      Angle_Order = angle_order_neac * (180 / 3.1415926535897932384626433832795L);  //conversion rad en deg pour affichage
      RudderDirectionOrder = rudder_direction_order;
      if (RudderDirectionOrder == 1) {
        // Mode = RudderDirectionOrder;
      }
      Serial.print("PILOTAGE ASPEDSA : ");
      Serial.println(rudder_direction_order);
      Serial.print("Angle Neac : ");
      Serial.println(Angle_Order);
    }
    Serial.println("Nikolaiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii");
    source = N2kMsg.Source;
    if (source == 28) {
      RudderPosition = rudder_position_nmea;
      RudFeedback = true;
    } else {
      RudFeedback = false;
    }


    //Serial.println(instance);
    //Serial.println(rudder_position);
    Serial.println(N2kMsg.Source);
  }
}

//-----------------------------------------------------------------------------------------------------
void Attitude(const tN2kMsg &N2kMsg) {
  unsigned char SID;
  double Yaw;
  double Pitch;
  double Roll;

  Serial.println("echooooooooooooooooooooooooooo");
  if (ParseN2kAttitude(N2kMsg, SID, Yaw, Pitch, Roll)) {


    Heading = (RAD_TO_DEG * Yaw);

    //Serial.print("yaw: ");
    //Serial.println(Yaw);
    //Serial.print("Heading: ");
    //Serial.println(Heading);
    //Serial.print("Pitch: ");
    //Serial.println(Pitch);
    //Serial.print("Roll: ");
    //Serial.println(Roll);
  } else {
#ifdef DEBUG_MODE
    ReadStream->print("Failed to parse PGN: ");
    ReadStream->println(N2kMsg.PGN);
#endif
  }
}

//-----------------------------------------------------------------------------------------------------

void HandleStreamN2kMsg(const tN2kMsg &N2kMsg) {
  int iHandler;
#ifdef DEBUG_MODE
  ForwardStream->println("_______________________________");
  ForwardStream->print(F("PGN (#"));
  printDouble(&Serial, nmea_received_id++, 10000);
  ForwardStream->print(F(") ID = "));
  ForwardStream->print(N2kMsg.PGN);
  ForwardStream->print(F(" - "));
#endif

  for (iHandler = 0; NMEA2000Handlers[iHandler].PGN != 0 && !(N2kMsg.PGN == NMEA2000Handlers[iHandler].PGN); iHandler++)
    ;
  if (NMEA2000Handlers[iHandler].PGN != 0) {
    NMEA2000Handlers[iHandler].Handler(N2kMsg);
  } else {
#ifdef DEBUG_MODE
    ForwardStream->println(F("Unknown PGN"));
#endif
  }
}

//-----------------------------------------------------------------------------------------------------
void setup() {

  // Init USB serial port
  Serial.begin(115200);  // Debug port
  /*
  * UART1  -> Serial1
  * RX Pin -> GPIO 14
  * TX Pin -> GPIO 12
  * UART Configuration -> SERIAL_8N1
  */
  Serial1.begin(115200, SERIAL_8N1, 14, 12);  // Messages with ASPEDSA PC

  Serial2.begin(115200);

  Wire.begin();
  // Init DAC MCP4725
  if (dac.begin() == false) {
    Serial.print("Connexion au module MCP4725 Gaz impossible, à l'adresse [0x");
    Serial.print(adresseI2cDacGaz, HEX);
    Serial.println("]");
  } else {
    Serial.println("Connexion au MCP4725 Gaz réussie !");
    is_DAC_init = 1;
  }
  // Init DAC thruster
  if (dac2.begin() == false) {
    Serial.print("Connexion au module dac thruster impossible, à l'adresse [0x");
    Serial.print(adresseI2cDacThruster, HEX);
    Serial.println("]");
  } else {
    Serial.println("Connexion au MCP4725 thruster réussie !");
    is_DAC2_init = 1;
  }

  // initialize with the I2C addr 0x3C
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  display.clearDisplay();
  // set text color
  //display.setTextColor(WHITE);
  display.setTextColor(SSD1306_BLACK, SSD1306_WHITE);
  // set text size
  display.setTextSize(2);
  // set text cursor position
  display.setCursor(0, 0);
  display.println("   NEAC   ");
  display.setCursor(0, 15);
  display.println(" AUTOPILOT ");
  //display.println("System startup");
  display.display();
  display.startscrollleft(0x00, 0x0F);
  delay(2000);
  display.stopscroll();

  display.clearDisplay();

  /* Serial to display data */
  // Serial.begin(115200);
  // while (!Serial) {}
  /* Begin the SBUS communication */
  sbus_rx.Begin();
  //sbus_tx.Begin();

  // Set product information
  NMEA2000.SetProductInformation("1",                       // Manufacturer's Model serial code
                                 100,                       // Manufacturer's product code
                                 "NMEA 2000 WiFi Gateway",  // Manufacturer's Model ID
                                 "1.0.2.25 (2019-07-07)",   // Manufacturer's Software version code
                                 "1.0.2.0 (2019-07-07)"     // Manufacturer's Model version
  );
  // Set device information
  NMEA2000.SetDeviceInformation(123459,  // Unique number. Use e.g. Serial number. Id is generated from MAC-Address
                                130,     // Device function=Analog to NMEA 2000 Gateway. See codes on http://www.nmea.org/Assets/20120726%20nmea%202000%20class%20&%20function%20codes%20v%202.00.pdf
                                25,      // Device class=Inter/Intranetwork Device. See codes on  http://www.nmea.org/Assets/20120726%20nmea%202000%20class%20&%20function%20codes%20v%202.00.pdf
                                2046     // Just choosen free from code list on http://www.nmea.org/Assets/20121020%20nmea%202000%20registration%20list.pdf
  );

  // If you also want to see all traffic on the bus use N2km_ListenAndNode instead of N2km_NodeOnly below
  //NMEA2000.SetForwardType(tNMEA2000::fwdt_Text);        // Show in clear text. Leave uncommented for default Actisense format.

  NMEA2000.SetForwardType(tNMEA2000::fwdt_Text);  // dpi
  NMEA2000.SetForwardStream(&Serial);
  NMEA2000.SetMsgHandler(HandleStreamN2kMsg);
  //NMEA2000.SetMode(tNMEA2000::N2km_ListenAndSend);
  NMEA2000.EnableForward(false);

  NMEA2000.SetMode(tNMEA2000::N2km_NodeOnly, 73);  // NodeAddress, default 32
  //NMEA2000.ExtendTransmitMessages(TransmitMessages);
  // NMEA2000.ExtendReceiveMessages(ReceiveMessages);
  NMEA2000.Open();
  NMEA2000.IsOpen();

  pinMode(RPWM, OUTPUT);
  pinMode(LPWM, OUTPUT);
  pinMode(R_EN, OUTPUT);
  pinMode(L_EN, OUTPUT);
  pinMode(UP, INPUT_PULLUP);
  pinMode(DOWN, INPUT_PULLUP);
  pinMode(VALID, INPUT_PULLUP);
  pinMode(ESCAPE, INPUT_PULLUP);
  pinMode(LEDGREEN, OUTPUT);
  //pinMode(LEDRED, OUTPUT);
  //pinMode(LEDYELL, OUTPUT);
  pinMode(LEDWHIT, OUTPUT);
  //pinMode(LEDBLUE, OUTPUT);
  pinMode(STARBORD, OUTPUT);
  pinMode(PORTSIDE, OUTPUT);
  //pinMode(JB, INPUT);
  //pinMode(JX, INPUT);
  //pinMode(JY, INPUT);
}

//-----------------------------------------------------------------------------------------------------
void Manage_Manette_Gaz() {
#ifdef DEBUG_RC_GAZ
  //Serial.println("------ ELEVATION = MANETTE DES GAZ  -----");
  Serial.print("  Elevation : ");
  Serial.println(gaz_value);
#endif
  if (killman > 1000) {
    if ((gaz_value > 170 && gaz_value < 811) || (gaz_value > 829 && gaz_value < 1812)) {
      //consigne_dac_gaz = gaz_value;
      valeurEnvoyeeAuDacGaz = map(gaz_value, 170, 1812, 480, 3790);  // 4095 pour sortie DAC=5V / 3790 pour limiter à 4,5V / 480 pour 0,5v
    }
    if (gaz_value > 810 && gaz_value < 830) {
      valeurEnvoyeeAuDacGaz = 2047;  // neutre
    }
  } else if (((Mode == 3) || (Mode == 5)) & throttle_order == 1) {
    valeurEnvoyeeAuDacGaz = map(throttle, -100, 100, 1, 4095);  // consigne Neac
  } else {
    valeurEnvoyeeAuDacGaz = 2047;  // neutre
    throttle_order = -1;
  }
  if (is_DAC_init == 1) {
    dac.writeDAC(valeurEnvoyeeAuDacGaz, false);  // Ecriture sur Digital Analog Converter, non sauvegardé en Eprom
  }

#ifdef DEBUG_RC_GAZ
  Serial.print("valeurEnvoyeeAuDacGaz : ");
  Serial.println(valeurEnvoyeeAuDacGaz);
#endif
}
//-----------------------------------------------------------------------------------------------------
void Manage_Commande_Barre() {
#ifdef DEBUG_RC_BARRE
  //Serial.println("------ AILERON = Tiller  -----");
  Serial.print("  Tiller : ");
  Serial.println(tiller_value);
#endif

  if ((Mode == 5) || (Mode == 6)) {
    if (Cap_Order < (Heading - 1)) {
      Ecart = (Heading - Cap_Order);
      if (Ecart > 20) {
        Ecart = 20;
      }
      Angle_Order = (Rudder_Position - Ecart);
    }
    if (Cap_Order > (Heading + 1)) {
      Ecart = (Cap_Order - Heading);
      if (Ecart > 20) {
        Ecart = 20;
      }
      Angle_Order = (Rudder_Position + Ecart);
    }
    if (Cap_Order == Heading) {
      Angle_Order = Rudder_Position;
    }
  }

  //Enable//

  //mode 1 RC
  if ((killman > 1000) & (RudFeedback == true)) {         // "Homme mort" radiocommande
    Angle_Order = map(tiller_value, 170, 1812, -45, 45);  // conversion voie radio vers angle tiller
    digitalWrite(R_EN, HIGH);
    digitalWrite(L_EN, HIGH);
    Serial.println("RC enable");
    Mode = 1;
  }

  //mode 2 N2K control
  else if ((Instance == 10) & (RudderDirectionOrder == 1) & (RudFeedback == true)) {
    digitalWrite(R_EN, HIGH);
    digitalWrite(L_EN, HIGH);
    Serial.println("NEAC 2K enable");
    Mode = 2;

  }
  //mode manuel local 4:FU, 6:HDG, 7:Set
  else if ((RudFeedback == true) & ((Mode == 4) || (Mode == 6) || ((Mode == 7) & (setA == 2)))) {
    digitalWrite(R_EN, HIGH);
    digitalWrite(L_EN, HIGH);
    Serial.println("LOCAL enable");
  }

  //mode 3 Neac Rudder Control
  else if (rudder_order == 0) {
    Angle_Order = rudder_value;
    digitalWrite(R_EN, HIGH);
    digitalWrite(L_EN, HIGH);
    Serial.println("NEAC enable mode 3 Rudder control");
    Mode = 3;
  }

  //mode 5 Neac Cap Control
  else if (rudder_order == 1) {
    digitalWrite(R_EN, HIGH);
    digitalWrite(L_EN, HIGH);
    Serial.println("NEAC enable mode 5 Cap control");
    Mode = 5;
  }

  //mode standby
  else {
    digitalWrite(R_EN, LOW);
    digitalWrite(L_EN, LOW);
    Serial.println("no enable");
    //Mode = 0;
  }

  //Steering//

  //tribord ROUGH
  if ((Angle_Order > (Rudder_Position + 10)) & (Rudder_Position < 50)) {
    digitalWrite(LPWM, LOW);
    //delay(10);
    analogWrite(RPWM, Speed);
    //Serial.println("tribord");
    if (Mode > 0) {
      display.setCursor(115, 25);
      display.println(">>");
    }
  }
  //tribord FINE
  if ((Angle_Order <= (Rudder_Position + 10)) & (Angle_Order >= (Rudder_Position + 1)) & (Rudder_Position < 55)) {
    digitalWrite(LPWM, LOW);
    //delay(10);
    analogWrite(RPWM, (Speed - 15));
    //Serial.println("tribord lent");
    if (Mode > 0) {
      display.setCursor(120, 25);
      display.println(">");
    }
  }
  //babord ROUGH
  if ((Angle_Order < (Rudder_Position - 10)) & (Rudder_Position > -50)) {
    digitalWrite(RPWM, LOW);
    //delay(10);
    analogWrite(LPWM, Speed);
    //Serial.println("babord");
    if (Mode > 0) {
      display.setCursor(0, 25);
      display.println("<<");
    }
  }
  //babord FINE
  if ((Angle_Order >= (Rudder_Position - 10)) & (Angle_Order <= (Rudder_Position - 1)) & (Rudder_Position > -55)) {
    digitalWrite(RPWM, LOW);
    //delay(10);
    analogWrite(LPWM, (Speed - 15));
    //Serial.println("babord lent");
    if (Mode > 0) {
      display.setCursor(0, 25);
      display.println("<");
    }
  }
  //Stop
  //if ((Rudder_Position <= -65) || ((Rudder_Position - 1) < Angle_Order) & (Angle_Order < (Rudder_Position + 1)) || (Rudder_Position > 65)) {
  if (((Angle_Order < Rudder_Position) & Rudder_Position <= -65) || (((Rudder_Position - 1) < Angle_Order) & (Angle_Order < (Rudder_Position + 1))) || ((Angle_Order > Rudder_Position) & (Rudder_Position > 65))) {
    digitalWrite(LPWM, LOW);
    digitalWrite(RPWM, LOW);
    //Serial.println("stop");
  }
}

//-------------------------------------------------------------------------------------------------
void Manage_Propulseur() {

#ifdef DEBUG_RC_PROPULSEUR
  //Serial.println("------ THRUSTER = PROPULSEUR d'ETRAVE  -----");
  Serial.print("  Thruster : ");
  Serial.println(thruster_value);
#endif

  if (thruster_value > 1820 && thruster_value < 2048) {
    thruster_value = 1811;
  }
  if (thruster_value < 160 && thruster_value > 0) {
    thruster_value = 171;
  }
  if (killman > 1000) {
    valeurEnvoyeeAuDacThruster = map(thruster_value, 171, 1811, 1, 4095);  // 4095 pour sortie DAC=5V
  } else if ((Mode == 3) || (Mode == 5)) {
    valeurEnvoyeeAuDacThruster = map(thruster, -100, 100, 1, 4095);  // consigne Neac
  } else {
    valeurEnvoyeeAuDacThruster = 2047;  // 4095 pour sortie maxi DAC=5V / 2047 pour neutre à 2,5V
  }

  if (is_DAC2_init == 1) {
    dac2.writeDAC(valeurEnvoyeeAuDacThruster, false);  // Ecriture sur Digital Analog Converter, non sauvegardé en Eprom
  }
#ifdef DEBUG_RC_PROPULSEUR
  Serial.print("valeurEnvoyeeAuDacThruster : ");
  Serial.println(valeurEnvoyeeAuDacThruster);
#endif
}

//-----------------------------------------------------------------------------------------------------
void boutonvalidation() {

  ETATBP2 = digitalRead(bouton2);                         // bouton # Enter
  if (((ETATBP2 == LOW) || (JoyB == 1)) & (setA == 0)) {  // si pas en mode "Set Angle" en cours
    Mode++;                                               // change de mode
  }
  Serial.print("Mode: ");
  Serial.println(Mode);

  if (Mode == 8) {  //selection du dernier mode revient au premier
    Mode = 0;
  }

  if (((ETATBP2 == LOW) || (JoyB == 1)) & (setA == 1)) {  // Mode 7: Set Angle Local
    Angle_Order = Set_Angle;                              // Execute Angle preset en local
    setA = 2;                                             // Set Angle Local validé
                                                          // Serial.println("VALIDATION");
    delay(50);
  }

  ETATBP4 = digitalRead(bouton4);  // bouton * Escape
  if (ETATBP4 == LOW) {
    Angle_Order = 0;           //reset Angle_Order
    comptage = 0;              //reset nombre incrémentation du Mode 7 Set Angle Local
    Set_Angle = 0;             // Mode 7 Set Angle Local défini
    setA = 0;                  // Mode 7 Set Angle Local desactivé
    Cap_Order == Heading;      // stock ordre de cap
    RudderDirectionOrder = 0;  // reset RudderDirectionOrder
    if (Mode != 7) {
      Mode = 0;  // retour Standby
    }
    // Serial.println("RESET");
    delay(100);
  }
  if ((ETATBP2 == LOW) && (ETATBP4 == LOW)) {
    Mode = 7;  // Forçage mode Set Angle Rudder
  }
}

void boutonUp() {
  ETATBP1 = digitalRead(bouton1);

  if (((ETATBP1 == LOW) || (JoyX < 10)) & (Mode < 1) & (Speed < 61)) {
    Speed = (Speed + 5);
  }

  if (((ETATBP1 == LOW) || (JoyX < 10)) & ((Mode == 3) || (Mode == 4)) & (Angle_Order < 45)) {  //bouton Up incrémente de 5° la barre depuis le mode 3 NeacFU: (Neac Rudder Order)
    Angle_Order += 5;
    Mode = 4;  //et force le Mode 4 FU: (Manual Rudder)
  }


  if (((ETATBP1 == LOW) || (JoyX < 10)) & ((Mode == 5) || (Mode == 6)) & (Cap_Order < 180)) {  //bouton Up incrémente de 5° la consigne de CAP depuis le mode 5 Neac HDG: (Neac Cap Order)
    Cap_Order += 5;
    Mode = 6;  //et force le Mode 6  HDG: (Manual Cap)
  }

  if (((ETATBP1 == LOW) || (JoyX < 10)) & (Mode == 7)) {  // Mode Set angle local
    comptage++;
    setA = 1;  // Mémoire Set in progress = 1
               // Serial.println("UP UP UP");
    delay(50);
    Set_Angle = comptage * 5;  //incrémente de 5
  }
}

void boutonDown() {
  ETATBP3 = digitalRead(bouton3);

  if (((ETATBP3 == LOW) || (JoyX > 1014)) & (Mode < 1) & (Speed > 29)) {
    Speed = (Speed - 5);
  }

  if (((ETATBP3 == LOW) || (JoyX > 1014)) & ((Mode == 3) || (Mode == 4)) & (Angle_Order > -45)) {  //bouton Down décrémente de 5° la barre depuis le mode 3 NeacFU: (Neac Rudder Order)
    Angle_Order -= 5;
    Mode = 4;  //et force le Mode 4 FU: (Manual Rudder)
  }

  if (((ETATBP3 == LOW) || (JoyX > 1014)) & ((Mode == 5) || (Mode == 6)) & (Cap_Order > -180)) {  //bouton Down décrémente de 5° la consigne de CAP depuis le mode 5 Neac HDG: (Neac Cap Order)
    Cap_Order -= 5;
    Mode = 6;  //et force le Mode 6  HDG: (Manual Cap)
  }

  if (((ETATBP3 == LOW) || (JoyX > 1014)) & (Mode == 7)) {  //Mode Set angle local
    comptage--;
    //  Serial.println("DOWN DOWN");
    setA = 1;
    delay(50);
    Set_Angle = comptage * 5;  //decrémente de 5
  }
}

// Permet de decouper une chaine de caractere puis de stocker dans un tableau
int splitString(String data2, char delimiter, String *result) {
  int index = 0;
  int start = 0;
  int end1 = data2.indexOf(delimiter);

  while (end1 != -1) {
    result[index++] = data2.substring(start, end1);
    start = end1 + 1;
    end1 = data2.indexOf(delimiter, start);
  }
  result[index++] = data2.substring(start);
  return index;
}

void aspedsaOrder(Stream &serialPort) {
  String result[7];
  String data2;

  // si utilisation d'un autre serial(exemple "Serial1" remplacer "Serial." par "Serial1." )
  if (serialPort.available() > 0) {
    // lecture jusqu'au caractere de fin chaine
    data2 = serialPort.readStringUntil('!');
    // decoupage puis recuperation du nombre de parametre recupere
    int index = splitString(data2, ';', result);
    // verifie si la chaine est complete
    if (index >= 6) {
      rudder_order = result[1].toInt();
      rudder_value = result[2].toDouble();
      throttle_order = result[3].toInt();
      throttle = result[4].toDouble();
      thruster = result[5].toInt();
      //serialPort.println(String(rudder_order) + "  " + String(rudder_value) + "  " + String(throttle_order) + "  " + String(throttle) + "  " + String(thruster));
    }
  }
}

//-----------------------------------------------------------------------------------------------------
void loop() {

  //digitalWrite(STARBORD, HIGH);
  //digitalWrite(PORTSIDE, HIGH);
  //digitalWrite(LEDGREEN, HIGH);
  analogWrite(LEDGREEN, 25);
  if (IsOpen == true) {
    Serial.print("IsOpen: ");
    Serial.println(IsOpen);
    //digitalWrite(LEDBLUE, 100);
  }


  //digitalWrite(LEDRED, HIGH);
  //digitalWrite(LEDYELL, HIGH);
  //digitalWrite(LEDWHIT, HIGH);
  //digitalWrite(LEDBLUE, HIGH);
  //
  //JoyB = digitalRead(JB);
  //Serial.print("JoyB : ");
  //Serial.println(JoyB);
  //JoyX = analogRead(JX);
  //Serial.print("JoyX : ");
  //Serial.println(JoyX);
  //JoyY = analogRead(JY);
  //Serial.print("JoyY : ");
  //Serial.println(JoyY);


  // Read sbus receiver
  if (sbus_rx.Read()) {
    /* Grab the received data */
    Serial.print("Etape 2");
    data = sbus_rx.data();
    /* Display the received data */
    Serial.println("Channel : ");
    for (int8_t i = 0; i < data.NUM_CH; i++) {
      Serial.print(data.ch[i]);
      Serial.print("\t");
    }

    tiller_value = data.ch[0];
    gaz_value = data.ch[1];
    mooring_value = data.ch[2];
    thruster_value = data.ch[3];
    rear = data.ch[4];
    modeRC = data.ch[5];
    killman = data.ch[6];
    tiller_offset = data.ch[7];

    Serial.print("killman ");
    Serial.println(killman);
    /* Display lost frames and failsafe data */
    Serial.print(data.lost_frame);
    Serial.print("\t");
    Serial.println(data.failsafe);
    Serial.println("Fin de channel ");
    /* Set the SBUS TX data to the received data */
    //sbus_tx.data(data);
    /* Write the data to the servos */
    //sbus_tx.Write();
    //digitalWrite(LEDYELL, 100);
    if (killman > 1000) {
      digitalWrite(LEDWHIT, HIGH);
    } else {
      digitalWrite(LEDWHIT, LOW);
    }
  }
  aspedsaOrder(Serial1);
  //void Rudder(const tN2kMsg &N2kMsg);
  //void Attitude(const tN2kMsg &N2kMsg);
  Manage_Commande_Barre();  // TILLER
  Manage_Manette_Gaz();     // GAZ
  Manage_Propulseur();      // Thruster

  boutonvalidation();  //fonction du bouton de validation du comptage led
  boutonUp();          //fonction incrémentation
  boutonDown();        //fonction décrémentation

  //Serial.print("comptage : ");
  //Serial.print(comptage);
  //Serial.println(" fois");


  //display.setCursor(1, 25);
  //display.println(".");
  //display.display();
  display.setTextColor(WHITE);
  display.setTextSize(1);  // set text size
  if (Mode > 3) {
    display.setCursor((Angle_Order + 60), 25);
    display.println("x");
  }

  if (RudFeedback == true) {
    display.setCursor((Rudder_Position + 60), 25);
    display.println("o");
  } else {
    display.setCursor((50), 25);
    display.println("no-rf");
  }

  display.display();
  display.clearDisplay();

  display.setTextColor(WHITE);  // set text color
  display.setTextSize(1);       // set text size
  display.setCursor(1, 0);      // set text cursor position
  if (Mode < 3) {
    display.print("Rudder speed:   ");
    display.print(Speed);
  } else if ((Mode == 5) || (Mode == 6)) {
    display.print("Cap: ");
    display.print(Cap_Order);
  } else if (Mode == 7) {
    display.print("Set Angle ");
  } else {
    display.print("Angle Order ");
  }

  if ((Mode == 7) & (setA == 1)) {
    display.setCursor(90, 0);  // set text cursor position
    display.setTextColor(SSD1306_BLACK, SSD1306_WHITE);
    display.println(Set_Angle);
  } else if ((Mode == 3) || (Mode == 4) || ((Mode == 7) & (setA != 1))) {
    display.setCursor(90, 0);  // set text cursor position
    display.setTextColor(WHITE);
    display.println(Angle_Order);
  } else if ((Mode == 5) || (Mode == 6)) {
    display.setCursor(70, 0);  // set text cursor position
    display.setTextColor(WHITE);
    display.print("Hdg: ");
    display.println(Heading);
  }

  display.setTextColor(WHITE);
  display.setCursor(1, 10);  // set text cursor position
  display.print("Rudder: ");
  display.setCursor(90, 10);  // set text cursor position
  display.println(Rudder_Position);

  display.setCursor(1, 20);  // set text cursor position
  display.print("Mode ");
  display.setCursor(30, 20);  // set text cursor position
  display.print(Mode);
  display.setCursor(45, 20);  // set text cursor position

  if (Mode == 3) {  // Neac Rudder // prise en main par l'ASPEDSA pour contrôle de barre
    display.print("NEAC FU");
  } else if (Mode == 4) {  // Manual Rudder // (re)prise en local d'une correction de barre
    display.print("    FU");
  } else if (Mode == 5) {  // NEAC CAP // prise en main par l'ASPEDSA pour contrôle de Cap
    display.print("NEAC HDG");
  } else if (Mode == 6) {  // Manual Cap  // (re)prise en local d'une correction de Cap
    display.print("   HDG");
  } else if (Mode == 7) {  // Set Manual Rudder // Set d'un angle de barre par Up and Down, validé par #, reset par *
    display.print("  PRESET");
  } else {
    display.print("Standby");
    //Angle_Order = 0;
  }

  // generate and clear display
  display.display();
  //display.clearDisplay();

  NMEA2000.ParseMessages();

  //Serial.print("set: ");
  //Serial.println(setA);
  //Serial.print("RudderDirectionOrder : ");
  //Serial.println(RudderDirectionOrder);

  //AngleOrder = Angle_Order * (3.1415926535897932384626433832795L / 180.0);  //conversion deg en rad pour affichage et execution

  /*brief Converting a value from Rad to Deg
 * \param   v   Input value in [rad]
 * \return      Corresponding value in [deg]
 */

  Serial.print("Angle Order: ");
  Serial.println(Angle_Order);

  Rudder_Position = RudderPosition * (180.0 / 3.1415926535897932384626433832795L);  //conversion rad en deg pour affichage et execution
  Serial.print("Rudder Position: ");
  Serial.println(Rudder_Position);



  //delay(10);
}
