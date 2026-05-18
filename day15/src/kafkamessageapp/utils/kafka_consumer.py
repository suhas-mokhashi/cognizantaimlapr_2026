#read messages from aiven kafka comedy-movies
from pathlib import Path

from kafka import KafkaConsumer, consumer
import json

from kafkamessageapp.configurations.conf import KafkaConfig

BASE_DIR = Path(__file__).resolve().parent.parent
def kafka_config():
    conf = {
        'bootstrap.servers': KafkaConfig.BOOTSTRAP_SERVERS,
        'security.protocol': KafkaConfig.SECURITY_PROTOCOL,
        'ssl.ca.location': str(BASE_DIR / "ca.pem"),
        'ssl.certificate.location': str(BASE_DIR / "service.cert"),
        'ssl.key.location': str(BASE_DIR / "service.key"),
         "group.id": "comedy-movies-consumer-group",
         "auto.offset.reset": "earliest",
    }
    return conf

def kafka_consumer():
    conf = kafka_config()
    consumer = KafkaConsumer(KafkaConfig.TOPIC_NAME, **conf)
    return consumer

def consume_messages(consumer):
    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None:
             continue

            if msg.error():
             print("Kafka error:", msg.error())
             continue

            data = json.loads(msg.value().decode("utf-8"))

      
            print(f"Received message: {data}") 


    except KeyboardInterrupt:
        print("Consumer stopped")

    finally:
        consumer.close()


if __name__ == "__main__":
    consumer = kafka_consumer()
    consume_messages(consumer)