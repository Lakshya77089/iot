
const int trigPin = 9;
const int echoPin = 10;
long duration;
int distance;
int te = 200;
void setup() {
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  Serial.begin(9600); 
}

void loop() {
  
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);


  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  if te > 

  // Read echoPin
  duration = pulseIn(echoPin, HIGH);

  // Distance in cm
  distance = duration * 0.034 / 2;

  // Print to Serial Monitor
  Serial.println(distance);
  delay(500);
}