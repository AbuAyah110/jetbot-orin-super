#!/usr/bin/env python3
"""PiOLED status on the Orin 40-pin I2C bus (JetPack 6.2: /dev/i2c-7)."""
from __future__ import annotations

import os
import socket
import time

import Adafruit_SSD1306
from PIL import Image, ImageDraw, ImageFont


OLED_BUS = int(os.environ.get("JETBOT_OLED_I2C_BUS", "7"))


def _iface_ip(name):
    path = "/sys/class/net/{0}/operstate".format(name)
    if not os.path.exists(path):
        return None
    try:
        if open(path).read().strip() == "down":
            return None
    except OSError:
        return None
    try:
        import fcntl
        import struct

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        packed = fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", name[:15].encode()))
        return socket.inet_ntoa(packed[20:24])
    except OSError:
        return None


def primary_ip():
    for name in ("wlP1p1s0", "wlan0", "wlan1", "eth0", "enP8p1s0"):
        ip = _iface_ip(name)
        if ip:
            return name, ip
    return None, None


def main():
    disp = Adafruit_SSD1306.SSD1306_128_32(rst=None, i2c_bus=OLED_BUS, gpio=1)
    disp.begin()
    disp.clear()
    disp.display()
    width, height = disp.width, disp.height
    image = Image.new("1", (width, height))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    host = socket.gethostname()

    while True:
        draw.rectangle((0, 0, width, height), outline=0, fill=0)
        ifname, ip = primary_ip()
        draw.text((0, 0), host[:21], font=font, fill=255)
        if ip:
            draw.text((0, 10), "{0}".format(ip), font=font, fill=255)
            draw.text((0, 20), ifname[:21], font=font, fill=255)
        else:
            draw.text((0, 12), "No IP yet", font=font, fill=255)
        disp.image(image)
        disp.display()
        time.sleep(2)


if __name__ == "__main__":
    main()
