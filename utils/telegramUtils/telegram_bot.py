import telegram_send

def telegram_send_message(message):
    try:
        telegram_send.send(messages=[message])
    except:
        print("NetworkError, telegram not reachable")