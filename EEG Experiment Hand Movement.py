from sqlite3 import Timestamp
from pygame.locals import *  # Just for some extra functions
import pygame
import datetime
import time

pygame.init() # Turns pygame 'on'

x = y = 0
running = 1
screen = pygame.display.set_mode((640, 400))
locations = []
with open("Try.txt", 'w') as f:
    while running:
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            running = 0
        elif event.type == pygame.MOUSEMOTION:
            ev_pos = event.pos
            locations.append((ev_pos,str(datetime.datetime.now())))
            f.write(str(locations))
            print("ev_pos[1]: " + str(ev_pos[1]) + "locations[0][0][1]: " + str(locations[0][0][1]) )


            def avg_time(min_date, max_date):
                min = time.mktime(datetime.datetime.strptime(min_date, '%y-%m-%d %H:%M:%S.%f').timetuple())
                max = time.mktime(datetime.datetime.strptime(max_date, '%y-%m-%d %H:%M:%S.%f').timetuple())
                t1 = Timestamp(min)
                t2 = Timestamp(max)
                av_time = Timestamp((t1.value + t2.value) / 2.0)
                return av_time

                f.write("movment found at: " + str(avg_time(locations[i][1], locations[i+1][1])) +"start loc: " + str(locations[i][0][1]) + "\n")
                locations.clear()
            if event.pos == (300,200):
                screen = pygame.display.set_mode((400, 500))
        screen.fill((0, 0, 0))
        pygame.display.flip()

f.close()