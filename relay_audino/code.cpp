// Define pins
int relay1 = 7;      // Relay for bulb
int relay2 = 8;      // Relay for motor
int pirSensor = 2;   // PIR sensor input pin

void setup() {
  pinMode(relay1, OUTPUT);
  pinMode(relay2, OUTPUT);
  pinMode(pirSensor, INPUT);

  // Turn both relays OFF initially (HIGH = OFF for active LOW relays)
  digitalWrite(relay1, HIGH);
  digitalWrite(relay2, HIGH);
}

void loop() {
  int pirState = digitalRead(pirSensor);

  if (pirState == HIGH) {
    // Motion detected: turn ON both relays
    digitalWrite(relay1, LOW);   // Relay1 ON (bulb)
    digitalWrite(relay2, LOW);   // Relay2 ON (motor)
  } else {
    // No motion: turn OFF both relays
    digitalWrite(relay1, HIGH);  // Relay1 OFF
    digitalWrite(relay2, HIGH);  // Relay2 OFF
  }

  delay(500); // Small delay to reduce bouncing
}
