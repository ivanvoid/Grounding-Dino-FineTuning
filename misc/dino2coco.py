import pandas as pd
import json

def convert_to_coco(df):
    # Initialize COCO structure
    coco_format = {
        "info": {
            "description": "Dataset description",
            "url": "http://example.com",
            "version": "1.0",
            "year": 2025,
            "contributor": "Ivan K",
            "date_created": "2025-06-09"
        },
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Define categories
    categories = {}
    category_id = 1

    # Process input data
    annotations = []
    image_ids = {}
    image_id_counter = 1
    annotation_id = 1

    # for line in input_data.strip().split('\n'):
    for index, row in df.iterrows():
        # parts = line.split(',')
        # import pdb;pdb.set_trace()
        class_name = row['label_name']
        x = int(row['bbox_x'])
        y = int(row['bbox_y'])
        width = int(row['bbox_width'])
        height = int(row['bbox_height'])
        image_file = row['image_name']
        img_width, img_height = map(int, [row['width'], row['height']])

        # Add image if not already added
        if image_file not in image_ids:
            coco_format["images"].append({
                "id": image_id_counter,
                "file_name": image_file,
                "width": img_width,
                "height": img_height
            })
            image_ids[image_file] = image_id_counter
            image_id_counter += 1

        # Get or create category
        if class_name not in categories:
            categories[class_name] = category_id
            coco_format["categories"].append({
                "id": category_id,
                "name": class_name
            })
            category_id += 1

        # Create annotation
        annotations.append({
            "id": annotation_id,
            "image_id": image_ids[image_file],
            "category_id": categories[class_name],
            "bbox": [x, y, width, height],
            "area": width * height,
            "iscrowd": 0
        })
        annotation_id += 1

    coco_format["annotations"] = annotations

    return coco_format


def main():
    # dataset_path = './multimodal-data/fashion_dataset_subset/val_annotations.csv'
    dataset_path = './multimodal-data/DINO_GAI20II/val_annotations.csv'

    df = pd.read_csv(dataset_path)

    print(df)
    # Convert to COCO format
    coco_data = convert_to_coco(df)

    # Save to JSON file
    with open('annotations.json', 'w') as json_file:
        json.dump(coco_data, json_file, indent=4)

    print("Conversion to COCO format completed and saved to 'annotations.json'.")


if __name__ == '__main__':
    main()