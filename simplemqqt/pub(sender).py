import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
PORT = 1883
TOPIC = "amitesh/test/topic"

client = mqtt.Client()
client.connect(BROKER, PORT, 60)

message = input("Enter message to send: ")

client.publish(TOPIC, message)
print(f"Message sent: {message}")

client.disconnect()
