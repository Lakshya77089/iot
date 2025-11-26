import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
PORT = 1883


mode = input("Choose mode (pub/sub): ").strip().lower()
session_type = input("Session type (clean/persistent): ").strip().lower()

TOPIC = "amitesh/mqtt/demo"

# Set session options
if session_type == "persistent":
    CLIENT_ID = "AmiteshPersistentClient"
    CLEAN = False
else:
    CLIENT_ID = ""
    CLEAN = True


def run_subscriber():
    client = mqtt.Client(client_id=CLIENT_ID, clean_session=CLEAN)

    def on_connect(client, userdata, flags, rc):
        print("\nConnected with code:", rc)
        print("Session Present:", flags.get("session present", "N/A"))
        print(f"Subscribing to topic: {TOPIC}")
        client.subscribe(TOPIC, qos=1)

    def on_message(client, userdata, msg):
        print(f"Message received: {msg.payload.decode()}")

    client.on_connect = on_connect
    client.on_message = on_message

    print("\nConnecting to broker...")
    client.connect(BROKER, PORT, 60)
    client.loop_forever()

def run_publisher():
    client = mqtt.Client(client_id=CLIENT_ID, clean_session=CLEAN)

    print("\nConnecting to broker...")
    client.connect(BROKER, PORT, 60)

    while True:
        msg = input("Enter message (or 'exit' to quit): ")
        if msg.lower() == "exit":
            break

        client.publish(TOPIC, msg, qos=1)
        print("Message sent.")

    client.disconnect()

if mode == "sub":
    run_subscriber()
elif mode == "pub":
    run_publisher()
else:
    print("Invalid mode. Choose 'pub' or 'sub'.")
