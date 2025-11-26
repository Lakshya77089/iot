// Final Code: LDR + PIR with Arduino (Blink LED in Dark when Motion Detected)

int ldrPin = A0;       // LDR connected to analog pin A0
int pirPin = 2;        // PIR sensor output connected to D2
int ledPin = 13;       // LED connected to D13
int ldrValue = 0;      // To store LDR reading
int pirStatus = 0;     // To store PIR reading
int threshold = 540;   // Threshold for darkness (adjust after testing)

void setup() {
  pinMode(pirPin, INPUT);
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  // Read LDR value
  ldrValue = analogRead(ldrPin);

  // Read PIR status
  pirStatus = digitalRead(pirPin);

  // Print values to Serial Monitor
  Serial.print("LDR Value: ");
  Serial.print(ldrValue);
  Serial.print(" | PIR: ");
  Serial.println(pirStatus == HIGH ? "Motion" : "No Motion");

  // Condition: Dark + Motion → Blink LED
  if (pirStatus == HIGH && ldrValue > threshold) {
    digitalWrite(ledPin, HIGH);
    delay(300);
    digitalWrite(ledPin, LOW);
    delay(300);
    Serial.println("Dark + Motion → LED Blinking");
  } else {
    digitalWrite(ledPin, LOW);  // Keep LED off otherwise
    Serial.println("LED OFF");
    delay(500);
  }
}
