import json
import cv2
import numpy as np
from time import sleep
from streamlink import Streamlink

from miio import Yeelight


config = json.load(open("config.json"))

class NeuroLamp:
    twitch_username = 'vedal987'

    x_start = config['x_start']
    x_end = config['x_end']
    y_start = config['y_start']
    y_end = config['y_end']

    lamp_ip = config['lamp_ip']
    lamp_token = config['lamp_token']

    last_color = np.array([0, 0, 0])

    update_interval = 1

    def __init__(self):
        self.stream_url = self.get_twitch_stream_url()
        self.yeelight = self.get_yeelight()


    def get_yeelight(self):
        return Yeelight(self.lamp_ip, self.lamp_token)


    def get_twitch_stream_url(self):
        """
        Get the best stream URL for the given Twitch channel using Streamlink's API.
        """
        session = Streamlink()
        try:
            streams = session.streams(f"https://www.twitch.tv/{self.twitch_username}")
            if "best" in streams:
                stream_url = streams["best"].url
                print(f"Stream URL: {stream_url}")
                return stream_url
            else:
                print("No 'best' stream available.")
                return None
        except Exception as e:
            print(f"Failed to get stream URL: {e}")
            return None


    # In GPT I trust
    def get_important_pixels(self, frame):
        """
        Find the brightest color and the color furthest from gray (excluding near black/white) in a rectangle at the bottom-right of the frame.
        """
        print(f"Saving debug frame.png")
        cv2.imwrite("frame.png", frame)

        roi = frame[self.y_start:self.y_end, self.x_start:self.x_end]

        print(f"Saving debug rectangle.png")
        cv2.imwrite("rectangle.png", roi)

        # Brightest color
        brightest_pixel = roi.reshape(-1, 3).max(axis=0)

        # Filter out black/white-like colors
        roi_reshaped = roi.reshape(-1, 3)
        brightness = np.linalg.norm(roi_reshaped, axis=1)  # Calculate brightness
        filtered_colors = roi_reshaped[(brightness > 150) & (brightness < 245)]

        if len(filtered_colors) > 0:
            # Calculate "distance from gray" for each color
            distances_from_gray = np.abs(filtered_colors[:, 0] - filtered_colors[:, 1]) + \
                                  np.abs(filtered_colors[:, 1] - filtered_colors[:, 2]) + \
                                  np.abs(filtered_colors[:, 2] - filtered_colors[:, 0])
            furthest_from_gray = filtered_colors[np.argmax(distances_from_gray)]
        else:
            # Default to previous color
            furthest_from_gray = self.last_color

        self.last_color = furthest_from_gray
        return brightest_pixel, furthest_from_gray, roi


    def get_lavalamp_color(self, frame):
        brightest, furthest_from_gray, roi = self.get_important_pixels(frame)

        brightest = [int(b) for b in brightest]
        brightness = sum(brightest) // 3 / 256 * 100
        brightness = int(brightness)
        # Values in array are reversed (bgr -> rgb)
        furthest_from_gray = [int(f) for f in furthest_from_gray[::-1]]

        return brightness, furthest_from_gray


    def run(self):
        while True:
            print("Capturing new frame")
            cap = cv2.VideoCapture(self.stream_url)
            if not cap.isOpened():
                print("Error: Unable to open the stream.")
                return

            ret, frame = cap.read()
            if ret:
                brightness, color = self.get_lavalamp_color(frame)
                self.update_lamp(brightness, color)

            else:
                print("Error: Unable to capture a frame.")

            # Release the video capture object
            cap.release()

            sleep(self.update_interval)


    def update_lamp(self, brightness, rgb):
        print(f"Brightness: {brightness}, RGB: {rgb}")
        if brightness < 1 or brightness > 100:
            print("Invalid brightness value")
            return
        if any([c < 1 or c > 255 for c in rgb]):
            print("Invalid RGB value, defaulting to brightness")
            rgb = [brightness] * 3
        self.yeelight.set_brightness(brightness)
        self.yeelight.set_rgb(rgb)



if __name__ == "__main__":
    lamp = NeuroLamp()
    lamp.run()
