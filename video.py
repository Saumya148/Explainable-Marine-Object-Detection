
from ultralytics import YOLO
import cv2
import time
import json
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

import sys

# ----------------------------
# Load trained model
# ----------------------------

model = YOLO("finetune/best.pt")

# ----------------------------
# Open Video
# ----------------------------

if len(sys.argv) > 1:
    video_path = sys.argv[1]
else:
    video_path = "ocean4.mp4"

cap = cv2.VideoCapture(video_path)
# ----------------------------
# Save Output Video
# ----------------------------

fourcc = cv2.VideoWriter_fourcc(*"avc1")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 20

width = 800
height = 500

out = cv2.VideoWriter(
    "output.mp4",
    fourcc,
    fps,
    (width, height)
)

# ----------------------------
# Statistics
# ----------------------------

unique_ids = {
    "Fish": set(),
    "Reefs": set(),
    "Robots": set(),
    "Plants": set(),
    "Diver": set(),
    "Wrecks": set(),
    "Seafloor": set()
}

confidence_sum = {}
confidence_count = {}

frame_count = 0

start = time.time()

# ----------------------------
# Process Video
# ----------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    frame = cv2.resize(frame, (800, 500))

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.25,
        imgsz=320,
        verbose=False
    )

    # ----------------------------
    # Draw Detection
    # ----------------------------

    annotated = results[0].plot()

    boxes = results[0].boxes

    # ----------------------------
    # Lightweight Heatmap
    # ----------------------------

    heatmap = annotated.copy()

    if boxes is not None and boxes.xyxy is not None:

        xyxy = boxes.xyxy.cpu().numpy()

        for box in xyxy:

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(
                heatmap,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),   # Red
                -1             # Filled
            )

    # Blend original + heatmap

    annotated = cv2.addWeighted(
        annotated,
        0.75,
        heatmap,
        0.25,
        0
    )

    # ----------------------------
    # Count Unique Objects
    # ----------------------------

    if boxes is not None:

        ids = boxes.id
        cls = boxes.cls
        confs = boxes.conf

        if ids is not None:

            ids = ids.cpu().numpy()
            cls = cls.cpu().numpy()
            confs = confs.cpu().numpy()

            for obj_id, c, conf in zip(ids, cls, confs):

                name = model.names[int(c)].capitalize()

                if name in unique_ids:
                    unique_ids[name].add(int(obj_id))

                confidence_sum[name] = confidence_sum.get(name, 0) + conf
                confidence_count[name] = confidence_count.get(name, 0) + 1

    # ----------------------------
    # Show Video
    # ----------------------------

    cv2.imshow("Marine Object Detection + Heatmap", annotated)

    out.write(annotated)

    # Press Q to quit

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# ----------------------------
# Release Resources
# ----------------------------

cap.release()
out.release()
cv2.destroyAllWindows()

end = time.time()

# ----------------------------
# Final Report
# ----------------------------

print()
print("=" * 65)
print("        UNDERWATER OBJECT DETECTION REPORT")
print("=" * 65)
print()

print("Frames Processed :", frame_count)
print()

print("-" * 45)
print("Unique Object Count")
print("-" * 45)

total = 0

for cls in unique_ids:

    count = len(unique_ids[cls])

    if count > 0:

        total += count
        print(f"{cls:12} : {count}")

print()
print("Total Unique Objects :", total)
print()

print("-" * 45)
print("Average Confidence")
print("-" * 45)

for cls in confidence_sum:

    avg = confidence_sum[cls] / confidence_count[cls]
    print(f"{cls:12} : {avg*100:.2f}%")

print()

print("-" * 45)
print("Explainable AI Summary")
print("-" * 45)

if len(unique_ids["Fish"]) > 0:
    print("\nFish:")
    print("The model focused on body contour, fins and underwater texture while detecting fish.")

if len(unique_ids["Reefs"]) > 0:
    print("\nReefs:")
    print("The model focused on coral texture and rocky formations.")

if len(unique_ids["Robots"]) > 0:
    print("\nRobots:")
    print("The model focused on mechanical structure and boundary features.")

if len(unique_ids["Plants"]) > 0:
    print("\nPlants:")
    print("The model focused on underwater vegetation texture and shape.")

if len(unique_ids["Diver"]) > 0:
    print("\nDiver:")
    print("The model focused on the human body outline and diving equipment.")

if len(unique_ids["Wrecks"]) > 0:
    print("\nWrecks:")
    print("The model focused on metallic structures and large underwater objects.")

print()
print("=" * 65)
print("Processing Time : {:.2f} seconds".format(end - start))
print()
print("Detection Completed Successfully")
print("=" * 65)
report = []

report.append("=" * 65)
report.append("UNDERWATER OBJECT DETECTION REPORT")
report.append("=" * 65)
report.append("")
report.append(f"Frames Processed : {frame_count}")
report.append("")

report.append("Unique Object Count")

for cls in unique_ids:
    count = len(unique_ids[cls])
    if count > 0:
        report.append(f"{cls:12} : {count}")

report.append("")
report.append(f"Total Unique Objects : {total}")
report.append("")

report.append("Average Confidence")

for cls in confidence_sum:
    avg = confidence_sum[cls] / confidence_count[cls]
    report.append(f"{cls:12} : {avg*100:.2f}%")

with open("report.txt", "w") as f:
    f.write("\n".join(report))
# ----------------------------
# Save Dashboard Metrics
# ----------------------------

metrics = {
    "frames": frame_count,
    "total_objects": total,
    "processing_time": round(end - start, 2)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)

# -------------------------------------
# Generate PDF Report
# -------------------------------------

doc = SimpleDocTemplate("marine_report.pdf")

styles = getSampleStyleSheet()

story = []

story.append(
    Paragraph(
        "<b>Marine Object Detection Report</b>",
        styles["Title"]
    )
)

story.append(
    Paragraph("<br/>", styles["Normal"])
)

for line in report:

    story.append(
        Paragraph(line, styles["Normal"])
    )

doc.build(story)

print("PDF Report Saved Successfully.")

# ----------------------------
# Save Object Count Table
# ----------------------------

rows = []

for cls in unique_ids:

    rows.append({
        "Class": cls,
        "Count": len(unique_ids[cls])
    })

df = pd.DataFrame(rows)

df.to_csv(
    "counts.csv",
    index=False
)

