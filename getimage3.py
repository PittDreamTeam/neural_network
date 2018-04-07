# python code for interfacing to VC0706 cameras and grabbing a photo
# pretty basic stuff
# written by ladyada. MIT license

import serial

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

s = serial.Serial(PORT, baudrate=BAUD, timeout=TIMEOUT)

def checkreply(r, b):
    r = list(map (lambda x: ord(x) if type(x) != int else x, r))
    if (r[0] == 0x76 and r[1] == SERIALNUM and r[2] == b and r[3] == 0x00):
        return True
    return False

def reset():
    cmd = ''.join (map (chr, resetcommand))
    s.write(cmd.encode())
    # print("reset", resetcommand)
    reply = s.read(100)
    r = list(reply)
    if checkreply(r, CMD_RESET):
        return True
    return False
        
def getversion():
    cmd = ''.join (map (chr, getversioncommand))
    s.write(cmd.encode())
    # print("getversion", getversioncommand)
    reply =  s.read(16)
    r = list(reply)
    if checkreply(r, CMD_GETVERSION):
        print("getversion reply: {}".format(r))
        return True
    return False

def takephoto():
    cmd = ''.join (map (chr, takephotocommand))
    s.write(cmd.encode())
    reply = s.read(5)
    r = list(reply)
    # print("Takephoto reply: {}".format(r))
    cr = checkreply(r, CMD_TAKEPHOTO)
    r3ez = r[3] == 0x0
    if cr and r3ez:
        return True
    return False

def getbufferlength():
    cmd = ''.join (map (chr, getbufflencommand))
    s.write(cmd.encode())
    # print("getbufferlength", getbufflencommand)
    reply = s.read(9)
    r = list(reply)
    print("Getbufferlen reply: {}".format(r))
    if (checkreply(r, CMD_GETBUFFLEN) and r[4] == 0x4):
        l = r[5]
        l <<= 8
        l += r[6]
        l <<= 8
        l += r[7]
        l <<= 8
        l += r[8]
        return l
    return 0

readphotocommand = [COMMANDSEND, SERIALNUM, CMD_READBUFF, 0x0c, FBUF_CURRENTFRAME, 0x0a]

def join_bytes(arr):
    res = b''
    for elem in arr:
        res += elem
    return res

def map_bytes(arr):
    return list(map(lambda x: x.to_bytes(1, 'big'), arr))

def readbuffer(num_bytes):
    addr = 0
    photo = []
    
    while (addr < num_bytes + 32):
        command = readphotocommand + [(addr >> 24) & 0xFF, (addr >> 16) & 0xFF,
                                      (addr >> 8) & 0xFF, addr & 0xFF]
        command +=  [0, 0, 0, 32]   # 32 bytes at a time
        command +=  [0, 0xff]       # delay of 10ms
        s.write(join_bytes(map_bytes(command)))
        # print("readcommand", command)
        reply = s.read(32+5+5)
        r = list(reply)
        if (len(r) != 37+5):
            print("Reply was wrong len: {}".format(len(r)))
            continue
        print("Readbuffer reply: {}".format(r))
        if (not checkreply(r, CMD_READBUFF)):
            print("ERROR READING PHOTO")
            exit()
        photo += r[5:-5]
        addr += 32
    return photo


def main():
    reset()

    if not getversion():
        print("Camera not found")
        exit()
    print("VC0706 Camera found")

    if takephoto():
        print("Snap!")

    num_bytes = getbufferlength()
    print(num_bytes, "bytes to read")
    photo = readbuffer(num_bytes)
    photodata = join_bytes(map_bytes(photo))
    print(type(photodata))
    with open('photo.jpg', 'wb') as f:
        f.write(photodata)

if __name__ == '__main__':
    main()