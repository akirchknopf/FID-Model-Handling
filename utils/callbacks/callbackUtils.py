from ..telegramUtils.telegram_bot import telegram_send_message

def plotTimesPerEpoch(callbacksList):
    minutesPerEpoch = []
    for time in callbacksList[3].times:
        minutesPerEpoch.append(int(round(time/60, 0)))
    telegram_send_message(f'It took so long to train on one epoch: {minutesPerEpoch} minutes')  
    print(f'It took so long to train on one epoch: {minutesPerEpoch} minutes')
    return f'It took so long to train on one epoch: {minutesPerEpoch} minutes'