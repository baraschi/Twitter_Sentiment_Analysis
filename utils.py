import sys, os, platform

if os.name == 'nt' and platform.release() == '10' and platform.version() >= '10.0.14393':
    # Fix ANSI color in Windows 10 version 10.0.14393 (Windows Anniversary Update)
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

'''Print by overriding the last line'''
def printOver(text):
    sys.stdout.write('\r' + text)
    sys.stdout.flush()
    # the big space string is necessary to really 'clean' the last lines as char remains otherwise... hacky but works..
