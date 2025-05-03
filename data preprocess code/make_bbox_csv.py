#!/usr/bin/env python
import pandas as pd
import numpy as np
import json

# Load the original CSV
df = pd.read_csv('affectnet.csv')

# Compute tight bbox (min/max over the 68 landmark points)
def compute_bbox(lm_json):
    pts = np.array(json.loads(lm_json)).reshape(-1, 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)

# Apply to every row
bboxes = df['landmarks'].apply(compute_bbox)
bbox_df = pd.DataFrame(bboxes.tolist(), columns=['x_min', 'y_min', 'x_max', 'y_max'])

# Assemble the new CSV
out = pd.DataFrame({
    'image_path': df['image_path'],
    'x_min':       bbox_df['x_min'],
    'y_min':       bbox_df['y_min'],
    'x_max':       bbox_df['x_max'],
    'y_max':       bbox_df['y_max'],
    'expression':  df['expression'].astype(int)
})

out.to_csv('affectnet_bbox.csv', index=False)
print(f"Wrote affectnet_bbox.csv with {len(out)} rows")

