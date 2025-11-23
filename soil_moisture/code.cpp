// Soil Moisture Sensor + Relay + LED with Arduino

int relayPin = 7;        // Relay control pin
int redLed = 8;          // LED indicator pin
int soilSensor = A0;     // Soil moisture sensor pin
int threshold = 300;     // Threshold value for dry soil

void setup()
{
  pinMode(redLed, OUTPUT);
  pinMode(relayPin, OUTPUT);
  pinMode(soilSensor, INPUT);

  Serial.begin(9600);   // For debugging & observation
  Serial.println("Soil Moisture Monitoring System Started...");
}

void loop()
{
  int moisture = analogRead(soilSensor);  // Read sensor value
  
  // Print the moisture value
  Serial.print("Soil Moisture Value: ");
  Serial.println(moisture);

  if (moisture <= threshold) {
    // Soil is dry → Turn ON pump & LED
    digitalWrite(redLed, HIGH);
    digitalWrite(relayPin, HIGH);
    Serial.println("Soil is Dry → Pump ON, LED ON");
  } 
  else {
    // Soil is wet → Turn OFF pump & LED
    digitalWrite(redLed, LOW);
    digitalWrite(relayPin, LOW);
    Serial.println("Soil is Wet → Pump OFF, LED OFF");
  }

  delay(10);  // Small delay for stable readings
}
