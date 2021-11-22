import os 
import sys 
sys.path.insert(0,os.getcwd())

from utils.utils import  get_rectangle_positions
from configs.camera import cfg as camera_config
if(__name__ == '__main__'):
    while(True):
        data = input()
        data = data.strip()
        data =data.split(" ")
        try:
            if(len(data) == 1):
                if(data[0] == 'e'):
                    break
            elif(len(data) == 2):
                x, y = data
                x,y = float(x), float(y)
                a_top = True 
            elif(len(data) == 3):
                x, y, a_top = data
                x,y, a_top= float(x), float(y), a_top == "True" or a_top == "t" or a_top == "true" or a_top == "T"
            else:
                print("wrong command")
        except:
            print("error")
        print(x, y, a_top)
        print("result::")
        print(get_rectangle_positions([x, y], camera_config))


    