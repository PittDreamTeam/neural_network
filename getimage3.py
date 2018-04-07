# python code for interfacing to VC0706 cameras and grabbing a photo
# pretty basic stuff
# written by ladyada. MIT license

import serial
import io
from PIL import Image

BAUD = 38400
PORT = "COM3"      # change this to your com port!
TIMEOUT = 0.2

SERIALNUM = 0    # start with 0

COMMANDSEND = 0x56
COMMANDREPLY = 0x76
COMMANDEND = 0x00

CMD_GETVERSION = 0x11
CMD_RESET = 0x26
CMD_TAKEPHOTO = 0x36
CMD_READBUFF = 0x32
CMD_GETBUFFLEN = 0x34

FBUF_CURRENTFRAME = 0x00
FBUF_NEXTFRAME = 0x01
FBUF_STOPCURRENTFRAME = 0x00

getversioncommand = [COMMANDSEND, SERIALNUM, CMD_GETVERSION, COMMANDEND]
resetcommand = [COMMANDSEND, SERIALNUM, CMD_RESET, COMMANDEND]
takephotocommand = [COMMANDSEND, SERIALNUM, CMD_TAKEPHOTO, 0x01, FBUF_STOPCURRENTFRAME]
getbufflencommand = [COMMANDSEND, SERIALNUM, CMD_GETBUFFLEN, 0x01, FBUF_CURRENTFRAME]
readphotocommand = [COMMANDSEND, SERIALNUM, CMD_READBUFF, 0x0c, FBUF_CURRENTFRAME, 0x0a]

def join_bytes(arr):
    res = b''
    for elem in arr:
        res += elem
    return res

def map_bytes(arr):
    return list(map(lambda x: x.to_bytes(1, 'big'), arr))

class Camera:
    def __init__(self):
        self.handle = serial.Serial(PORT, baudrate=BAUD, timeout=TIMEOUT)
        self.reset()
        if not self.getversion():
            raise Exception("Could not find camera")
        

    def checkreply(self, r, b):
        r = list(map (lambda x: ord(x) if type(x) != int else x, r))
        if (r[0] == 0x76 and r[1] == SERIALNUM and r[2] == b and r[3] == 0x00):
            return True
        return False

    def reset(self):
        cmd = ''.join (map (chr, resetcommand))
        self.handle.write(cmd.encode())
        # print("reset", resetcommand)
        reply = self.handle.read(100)
        r = list(reply)
        if self.checkreply(r, CMD_RESET):
            return True
        return False
            
    def getversion(self):
        cmd = ''.join (map (chr, getversioncommand))
        self.handle.write(cmd.encode())
        # print("getversion", getversioncommand)
        reply =  self.handle.read(16)
        r = list(reply)
        if self.checkreply(r, CMD_GETVERSION):
            print("getversion reply: {}".format(r))
            return True
        return False

    def takephoto(self):
        cmd = ''.join (map (chr, takephotocommand))
        self.handle.write(cmd.encode())
        reply = self.handle.read(5)
        r = list(reply)
        # print("Takephoto reply: {}".format(r))
        cr = self.checkreply(r, CMD_TAKEPHOTO)
        r3ez = r[3] == 0x0
        if cr and r3ez:
            return True
        return False

    def getbufferlength(self):
        cmd = ''.join (map (chr, getbufflencommand))
        self.handle.write(cmd.encode())
        # print("getbufferlength", getbufflencommand)
        reply = self.handle.read(9)
        r = list(reply)
        print("Getbufferlen reply: {}".format(r))
        if (self.checkreply(r, CMD_GETBUFFLEN) and r[4] == 0x4):
            l = r[5]
            l <<= 8
            l += r[6]
            l <<= 8
            l += r[7]
            l <<= 8
            l += r[8]
            return l
        return 0

    def readbuffer(self, num_bytes):
        addr = 0
        photo = []
        
        while (addr < num_bytes + 32):
            command = readphotocommand + [(addr >> 24) & 0xFF, (addr >> 16) & 0xFF,
                                        (addr >> 8) & 0xFF, addr & 0xFF]
            command +=  [0, 0, 0, 32]   # 32 bytes at a time
            command +=  [0, 0xff]       # delay of 10ms
            self.handle.write(join_bytes(map_bytes(command)))
            # print("readcommand", command)
            reply = self.handle.read(32+5+5)
            r = list(reply)
            if (len(r) != 37+5):
                print("Reply was wrong len: {}".format(len(r)))
                continue
            print("Readbuffer reply: {}".format(r))
            if (not self.checkreply(r, CMD_READBUFF)):
                print("ERROR READING PHOTO")
                exit()
            photo += r[5:-5]
            addr += 32
        return photo

    def take_photo(self):
        if not self.takephoto():
            raise Exception("Unable to take photo!")

        num_bytes = self.getbufferlength()
        print(num_bytes, "bytes to read")
        photo = self.readbuffer(num_bytes)
        photodata = join_bytes(map_bytes(photo))
        return Image.open(io.BytesIO(photodata))

if __name__ == '__main__':
    cam = Camera()
    cam.take_photo().show()
