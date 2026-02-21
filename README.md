# LightOCRX

LightOCRX is a lightweight C++ inference engine for PaddleOCR ONNX models (v4 and v5). It prioritizes speed, low memory footprint, and portability, making it ideal for low-end devices.

---

## This program depends on:

- libharu (PDF library)
- onnxruntime (ONNX model inference runtime)
- C++17 compiler

Please install both libraries before compiling.

## Usage

```bash
lightocrx <detection_model.onnx> <recognition_model.onnx> <images...> [options]
```

### Arguments

- `detection_model.onnx`  
  Path to detection model

- `recognition_model.onnx`  
  Path to recognition model

- `images...`  
  One or more PNG/JPG image files or a directory

---

## Options

- `-boxt`  
  Box confidence threshold (default: 0.6)

- `-bint`  
  Binarization threshold (default: 0.3)

- `-unclip`  
  Unclip ratio (default: 2.0)

- `-targetsize <int>`  
  Target size for detection (default: 640).  
  Higher values improve accuracy at the cost of performance.

- `-savecrops <0/1>`  
  Save cropped text regions (default: 0).  
  Creates a folder named `cropped` containing cropped PNG images.

- `-debugboxes <0/1>`  
  Draw detected boxes on output image (default: 0)

- `-debugout <path>`  
  Output path for debug image (default: detected_boxes.png)

- `-dict <path>`  
  Path to dictionary file (default: dict.txt)

- `-targetw <int>`  
  Target width for recognition (default: 0)  
  `0` = dynamic  
  `-1` = use crop aspect ratio

- `-targeth <int>`  
  Target height for recognition (default: 48)

- `-pdf <path>`  
  Export as searchable PDF (optional).  
  Creates a multi-page PDF when multiple images are provided.

- `-minconf <float>`  
  Minimum confidence for PDF export (default: 0.3)

- `-pdftext`  
  Export PDF as text-only (white background, black visible text, no images)

- `-text <path>`  
  Export recognized text to a plain text file

- `-textcls <0/1>`  
  Enable/disable text orientation classification and 180° rotation correction (default: 0).  
  Useful when images contain rotated text.

- `-clsmodel <path>`  
  Path to rotation classification model (required if `-textcls 1`)

- `-h`, `--help`  
  Show help message



## Examples

### Basic usage:

lightocrx det.onnx rec.onnx image1.jpg image2.jpg -boxt 0.2 -dict mydict.txt

### Save crops and debug boxes:

lightocrx det.onnx rec.onnx *.jpg -savecrops 1 -debugboxes 1

### Process all images in a directory and export PDF:

lightocrx det.onnx rec.onnx images/ -pdf output.pdf

### Export text-only PDF:

lightocrx det.onnx rec.onnx image1.jpg image2.jpg -pdf output.pdf -pdftext

### Export recognized text to file:

lightocrx det.onnx rec.onnx image1.jpg image2.jpg -text output.txt


### Enable rotation correction:

lightocrx det.onnx rec.onnx image.jpg -textcls 1 -clsmodel cls.onnx
```

