import os

def createDirIfNotExists(dirToCheck):
    if not os.path.exists(dirToCheck):
        print(f'Directory {dirToCheck} does not exist, creating it instead!')
        os.makedirs(dirToCheck)

def writeLog(pathToLogFile, message):
    logFile = open(pathToLogFile,"a+") 
    logFile.write(message + "\n")
    logFile.close()