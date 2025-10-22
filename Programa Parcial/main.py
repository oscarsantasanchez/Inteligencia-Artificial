# main.py
from detector import Detector

if __name__ == "__main__":
    detector = Detector(cam_index=0, debug=True)
    detector.run()
