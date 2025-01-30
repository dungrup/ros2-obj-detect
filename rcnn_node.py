import torch
import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from torchvision import models
import numpy as np
import time

labels_dict = { 0 : 'UNKNOWN', 1 : 'VEHICLE', 2 : 'PEDESTRIAN', 3 : 'SIGN', 4:'CYCLIST'}
label_threshold = 0.8
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_model():
    n_classes = 5

    # lets load the faster rcnn model
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)
    return model

model = get_model()
ckpt_file = 'frcnn_final_rr.pth'
model.load_state_dict(torch.load(ckpt_file))
model.to(device)
model.eval()

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'camera/image',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Image, 'camera/od_img', 10)
        self.bridge = CvBridge()
        self.vid_writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))


    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.get_logger().info('Receiving image frame')


        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(np.array(image).copy())
        image = image.permute(2, 0, 1)
        image = image / 255.0
        image = image.to(device)
        image = [image]
        with torch.no_grad():
                start = time.time()
                predictions = model(image)
                print(f"Time taken for inference: {(time.time() - start)* 1000.0} ms")

        for i, pred in enumerate(predictions):
                # Filter out predictions below the threshold
            scores = pred['scores']
            valid_indices = scores > label_threshold
            valid_boxes = pred['boxes'][valid_indices].cpu()
            valid_labels = pred['labels'][valid_indices].cpu()
            valid_scores = scores[valid_indices].cpu()
            
            # Draw bounding boxes and labels on the image
            for label, box in zip(valid_labels, valid_boxes):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, labels_dict[label.item()], (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Save the image with bounding boxes
        self.vid_writer.write(frame)
        self.get_logger().info('Writing image frame')
        self.publisher.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()

    rclpy.spin(image_subscriber)
    image_subscriber.vid_writer.release()
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
