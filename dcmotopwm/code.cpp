const int motorPWM = 9;
const int motorIn1 = 8;
const int motorIn2 = 7;

int pwmValues[] = {0, 50, 100, 150, 200, 255};
const int maxRPM = 3000;

void setup() {
  pinMode(motorPWM, OUTPUT);
  pinMode(motorIn1, OUTPUT);
  pinMode(motorIn2, OUTPUT);
  Serial.begin(9600);

  Serial.println("PWM\tDuty(%)\tRPM\t\tBehavior\tDirection");
  Serial.println("-----------------------------------------------------------");
}

void loop() {
  runMotor("Clockwise");
  runMotor("Anticlockwise");
}

void runMotor(String direction) {
  if (direction == "Clockwise") {
    digitalWrite(motorIn1, HIGH);
    digitalWrite(motorIn2, LOW);
  } else {
    digitalWrite(motorIn1, LOW);
    digitalWrite(motorIn2, HIGH);
  }

  for (int i = 0; i < 6; i++) {
    int pwm = pwmValues[i];
    float duty = (pwm / 255.0) * 100;
    float rpm = (duty / 100.0) * maxRPM;
    String behavior = (pwm > 0) ? "ON" : "OFF";

    analogWrite(motorPWM, pwm);
    delay(1000);
    printRow(pwm, duty, rpm, behavior, direction);
  }
}

void printRow(int pwm, float duty, float rpm, String behavior, String direction) {
  Serial.print(pwm);
  Serial.print("\t");

  if (duty < 10) Serial.print(" ");
  Serial.print(duty, 1);
  Serial.print("%\t");

  if (rpm < 1000) Serial.print(" ");
  Serial.print(rpm, 0);
  Serial.print(" RPM\t");

  Serial.print(behavior);
  Serial.print("\t\t");

  Serial.println(direction);
}
