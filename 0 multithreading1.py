import cv2
import threading

class CameraThread(threading.Thread):
    def __init__(self, index, name):
        threading.Thread.__init__(self)
        self.index = index
        self.name = name
        self.stopped = False
        self.frame = None

    def run(self):
        self.cap = cv2.VideoCapture(self.index)
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
        self.cap.release()

    def stop(self):
        self.stopped = True

def main():
    # Define camera indices
    camera_indices = {
        'Camera 1': 0,  # Index for Camera 1
        'Camera 2': 1,  # Index for Camera 2
        'Camera 3': 2   # Index for Camera 3
    }

    # Initialize threads
    threads = []
    for name, index in camera_indices.items():
        t = CameraThread(index, name)
        threads.append(t)
        t.start()

    while True:
        for t in threads:
            if t.frame is not None:
                cv2.imshow(t.name, t.frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Stop all threads
    for t in threads:
        t.stop()
        t.join()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
