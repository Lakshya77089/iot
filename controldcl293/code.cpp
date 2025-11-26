int motorPin1 = 2;   // Motor driver IN1
int motorPin2 = 3;   // Motor driver IN2

void setup() {
  pinMode(motorPin1, OUTPUT);
  pinMode(motorPin2, OUTPUT);

  Serial.begin(9600);   // Start serial communication at 9600 baud
  Serial.println("DC Motor Control Started");
}

void loop() {
  // Rotate Clockwise
  Serial.println("Motor rotating CLOCKWISE");
  digitalWrite(motorPin1, HIGH);
  digitalWrite(motorPin2, LOW);
  delay(2000);   // Rotate for 2 seconds

  // Stop
  Serial.println("Motor STOPPED");
  digitalWrite(motorPin1, LOW);
  digitalWrite(motorPin2, LOW);
  delay(1000);   // Pause for 1 sec

  // Rotate Anticlockwise
  Serial.println("Motor rotating ANTICLOCKWISE");
  digitalWrite(motorPin1, LOW);
  digitalWrite(motorPin2, HIGH);
  delay(2000);   // Rotate for 2 seconds

  // Stop
  Serial.println("Motor STOPPED");
  digitalWrite(motorPin1, LOW);
  digitalWrite(motorPin2, LOW);
  delay(1000);   // Pause for 1 sec
}