# This branch

This adds alpha control for fine-control of KL divergence.

# Datasets
HPDv2 train: ymhao/HPDv2
HPDv2 test: zhwang/HPDv2

HPDv2 train format:

```
DatasetDict({
    train: Dataset({
        features: ['prompt', 'image_path', 'raw_annotations', 'user_hash', 'image', 'rank', 'human_preference'],
        num_rows: 645090
    })
    test: Dataset({
        features: ['prompt', 'image_path', 'raw_annotations', 'user_hash', 'image', 'rank', 'human_preference'],
        num_rows: 400
    })
})
```

```json
{
  'prompt': 'A valley filled with red flowers with smoke flowing around - digital art by Studio Ghibli.',
  'image_path': "['00000000.jpg', '00000001.jpg']",
  'raw_annotations': [],
  'user_hash': [],
  'image': [<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=480x480 at 0x7F543F7F4490>, <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x7F543F7F4340>],
  'rank': [],
  'human_preference': [0, 1]
}
```
