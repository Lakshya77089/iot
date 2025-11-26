String inputString = "";

void setup() {
  Serial.begin(9600);
  pinMode(8, OUTPUT);   // light
  pinMode(9, OUTPUT);   // Fan
  Serial.println("Enter commands: LON, LOFF, FON, FOFF");
}

void loop() {
  if (Serial.available() > 0) {
    inputString = Serial.readString();
    inputString.trim();

    if (inputString == "LON") {
      digitalWrite(8, HIGH);
      Serial.println("Light ON");
    }
    else if (inputString == "LOFF") {
      digitalWrite(8, LOW);
      Serial.println("Light OFF");
    }
    else if (inputString == "FON") {
      digitalWrite(9, HIGH);
      Serial.println("Fan ON");
    }
    else if (inputString == "FOFF") {
      digitalWrite(9, LOW);
      Serial.println("Fan OFF");
    }
    else {
      Serial.println("Invalid Command");
    }
  }
}
