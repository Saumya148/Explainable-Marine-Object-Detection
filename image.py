
from ultralytics import YOLO
import cv2

# Load model
model = YOLO("finetune/best.pt")

# Read image
image = cv2.imread("test3.jpg")

# Run prediction
results = model(image)

# Draw detections
annotated_image = results[0].plot()

# Show image
cv2.imshow("Marine Detection", annotated_image)

# Wait until key press
cv2.waitKey(0)

# Close window
cv2.destroyAllWindows()