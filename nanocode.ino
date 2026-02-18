#define RED_LED    4
#define BLUE_LED   5
#define YELLOW_LED 6   // Yellow LED acts as buzzer indicator

void setup() {
  Serial.begin(9600);
  pinMode(RED_LED,    OUTPUT);
  pinMode(BLUE_LED,   OUTPUT);
  pinMode(YELLOW_LED, OUTPUT);
  digitalWrite(RED_LED,    LOW);
  digitalWrite(BLUE_LED,   LOW);
  digitalWrite(YELLOW_LED, LOW);
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "CRITICAL") {
      // Red ON + Yellow ON (buzzer) + Blue OFF
      digitalWrite(RED_LED,    HIGH);
      digitalWrite(YELLOW_LED, HIGH);
      digitalWrite(BLUE_LED,   LOW);
    }
    else if (cmd == "SLEEP") {
      // Red ON + Blue ON + Yellow ON (buzzer)
      digitalWrite(RED_LED,    HIGH);
      digitalWrite(BLUE_LED,   HIGH);
      digitalWrite(YELLOW_LED, HIGH);
    }
    else if (cmd == "NORMAL") {
      // All OFF
      digitalWrite(RED_LED,    LOW);
      digitalWrite(BLUE_LED,   LOW);
      digitalWrite(YELLOW_LED, LOW);
    }
  }
}