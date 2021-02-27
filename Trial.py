import time

fpsLimit = 1 # throttle limit
startTime = time.time()
nowTime = time.time()
print(startTime)
while True:

    nowTime = time.time()
    if (int(nowTime - startTime)) > fpsLimit:
        print(nowTime)
        startTime = time.time()
