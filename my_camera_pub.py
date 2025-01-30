import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'camera/image', 10)
        self.timer = self.create_timer(1/30, self.timer_callback)  # Set to 30Hz
        self.cap = cv2.VideoCapture(0)
        self.bridge = CvBridge()

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error('Failed to capture image')
            return

        msg = Image()
        msg.height, msg.width, _ = frame.shape
        msg.encoding = 'bgr8'
        msg.step = msg.width * 3
        msg.data = self.bridge.cv2_to_imgmsg(frame, 'bgr8').data

        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing image frame: height={msg.height}, width={msg.width}')

def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()
    rclpy.spin(camera_publisher)
    camera_publisher.cap.release()
    camera_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()