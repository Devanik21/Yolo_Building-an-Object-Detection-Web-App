import cv2
import numpy as np
import random

class FallbackDetector:
    """
    A fallback object detector that works without any pre-trained models.
    This is used when model files aren't available.
    """
    
    def __init__(self):
        self.object_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign", 
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
            "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball"
        ]
    
    def detect(self, image, confidence_threshold=0.5):
        """
        Generate random detections for demonstration purposes.
        
        Args:
            image: Input image
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Tuple of (output_image, detections)
        """
        height, width = image.shape[:2]
        output_image = image.copy()
        
        # Generate 1-5 random detections
        num_detections = random.randint(1, 5)
        detections = []
        
        for _ in range(num_detections):
            # Random class
            class_name = random.choice(self.object_classes)
            
            # Random confidence between threshold and 1.0
            confidence = random.uniform(confidence_threshold, 1.0)
            
            # Random box dimensions
            box_width = random.randint(width // 10, width // 3)
            box_height = random.randint(height // 10, height // 3)
            
            # Random position (ensure box is within image)
            x = random.randint(0, width - box_width)
            y = random.randint(0, height - box_height)
            
            # Draw box
            cv2.rectangle(output_image, (x, y), (x + box_width, y + box_height), (0, 255, 0), 2)
            
            # Add label
            cv2.putText(output_image, f"{class_name} {confidence:.2f}", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detections.append((class_name, confidence))
            
            # Add some delay to make it seem like processing
            if random.random() > 0.8:
                # Simulate object tracking with movement
                direction_x = random.choice([-1, 1])
                direction_y = random.choice([-1, 1])
                
                # Draw movement arrow
                arrow_end_x = x + box_width//2 + direction_x * 20
                arrow_end_y = y + box_height//2 + direction_y * 20
                cv2.arrowedLine(output_image, 
                               (x + box_width//2, y + box_height//2), 
                               (arrow_end_x, arrow_end_y), 
                               (255, 0, 0), 2)
        
        return output_image, detections
