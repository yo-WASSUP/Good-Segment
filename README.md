# Good-Segment

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)](https://opencv.org/)

**Make Image Segmentation Simple and Efficient!**

Interactive and automated image segmentation toolkit based on MobileSAM  
Batch Processing | Interactive Operation | Ready to Use

[Features](#-features) • [Quick Start](#-installation) • [Usage](#-usage) • [Showcase](#-showcase)

[中文文档](README_CN.md)

</div>

---

## 📷 Showcase

<div align="center">

### Tool Interface
![Tool Showcase](docs/examples/showcase.png)

### Segmentation Results
<table>
  <tr>
    <td align="center"><b>Original Image</b></td>
    <td align="center"><b>Generated Mask</b></td>
  </tr>
  <tr>
    <td><img src="docs/examples/000000.png" width="400"/></td>
    <td><img src="docs/examples/mask_000000.png" width="400"/></td>
  </tr>
</table>

</div>

## ✨ Features

This project provides three different image segmentation tools:

### 1. Interactive Batch Segmentation (`batch_mask_interactive.py`)
- 📦 **Batch Processing**: Process images one by one from a folder
- 🖱️ **Manual Box Drawing**: Draw one or multiple boxes for each image
- 🎯 **Precise Control**: Full control over segmentation areas
- 💾 **Auto Save**: Automatically save generated masks

**Use Case**: When you need to precisely specify segmentation areas for each image

### 2. Auto Batch Segmentation (`batch_mask.py`)
- ⚡ **Fully Automatic**: No manual intervention required
- 🎛️ **Multiple Modes**:
  - Center point mode
  - Grid point mode (3×3)
  - Full image box mode
  - Center region box mode
  - Custom relative coordinate box mode
- 🚀 **High Efficiency**: Suitable for large-scale batch processing

**Use Case**: Batch processing of images with similar composition

### 3. Single Image Interactive Segmentation (`interactive_mask.py`)
- 🎨 **Point Mode**: Segment by clicking foreground/background points
- 📐 **Box Mode**: Segment by drawing rectangular boxes
- 🔄 **Mode Switching**: Support mixed use of points and boxes
- 👀 **Real-time Preview**: Instant view of segmentation results

**Use Case**: Testing effects, single image fine segmentation

## 📋 Requirements

- Python 3.8+
- CUDA (Optional, for GPU acceleration)

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yo-WASSUP/Good-Segment.git
cd Good-Segment
```

### 2. Install Dependencies

```bash
pip install opencv-python
pip install numpy
pip install ultralytics
```

Or use requirements.txt:

```bash
pip install -r requirements.txt
```

### 3. Model File

✅ **Model Included**: This project includes the `mobile_sam.pt` model file. You can use it directly after cloning, no additional download required.

> 💡 **Model Information**:
> - Filename: `mobile_sam.pt`
> - Location: Project root directory
> - Source: [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) - Lightweight Segment Anything Model

## 🚀 Usage

### Method 1: Interactive Batch Segmentation (Recommended)

Manually select boxes for each image, suitable for scenarios requiring precise control.

```bash
# Basic usage
python batch_mask_interactive.py images/test

# Specify output directory
python batch_mask_interactive.py images/test -o output/masks

# Specify model path
python batch_mask_interactive.py images/test -m path/to/mobile_sam.pt
```

**Controls:**
- 🖱️ **Drag Mouse**: Draw rectangular box (multiple boxes supported)
- ⌨️ **Space Key**: Generate mask and move to next image
- ⌨️ **S Key**: Skip current image
- ⌨️ **R Key**: Reset boxes for current image
- ⌨️ **Q Key**: Quit program

**Color Indicators:**
- 🟣 Purple Box: Box being drawn
- 🟢 Green Box: Completed box

### Method 2: Auto Batch Segmentation

Fully automatic processing, suitable for batch processing similar images.

```bash
# Run the program
python batch_mask.py
```

Select mode according to prompts:
1. **Auto Mode - Center Point**: Use image center point as prompt
2. **Auto Mode - Grid Points**: Use 3×3 grid points
3. **Box Mode - Full Image**: Use entire image as box
4. **Box Mode - Center 80%**: Use center 80% region
5. **Box Mode - Custom**: Enter relative coordinates (0-1)

### Method 3: Single Image Interactive Segmentation

Suitable for testing effects and fine segmentation of single images.

```bash
python interactive_mask.py
```

**Controls:**
- **Point Mode**:
  - Left Click: Add foreground point (green)
  - Right Click: Add background point (red)
- **Box Mode**:
  - Drag: Draw rectangular box (purple)
- **General Operations**:
  - Space Key: Generate mask
  - M Key: Switch between point/box mode
  - R Key: Reset all points and boxes
  - Q Key: Quit program

> 💡 **Tip**: You can use points and boxes together for more precise segmentation!

## 📁 Project Structure

```
Good-Segment/
├── batch_mask_interactive.py   # Interactive batch segmentation tool
├── batch_mask.py               # Auto batch segmentation tool
├── interactive_mask.py         # Single image interactive segmentation tool
├── mobile_sam.pt               # MobileSAM model file (included)
├── requirements.txt            # Python dependencies
├── docs/                       # Documentation and examples
│   └── examples/               # Showcase images
├── images/                     # Input image directory (example)
│   └── test/                   # Test images
├── output/                     # Output directory (auto-created)
│   └── masks/                  # Generated masks
├── .gitignore                  # Git ignore file
└── README.md                   # Project documentation
```

## 📖 Output Format

All tools generate masks in the following format:
- **Format**: PNG image
- **Type**: Single-channel binary image
- **Values**:
  - White (255): Foreground/Object
  - Black (0): Background

## 🎯 Use Cases

| Tool | Use Case | Advantages | Disadvantages |
|------|----------|------------|---------------|
| Interactive Batch | Medium quantity of images requiring precise segmentation | Precise control, multi-box support | Requires manual operation |
| Auto Batch | Large quantity of similar composition images | Fast, fully automatic | May need parameter adjustment |
| Single Image Interactive | Testing effects, single image fine segmentation | High flexibility, point+box support | Only processes one image |

## ⚠️ Notes

1. **Model File**: Project includes `mobile_sam.pt` model file, ready to use after cloning
2. **Image Format**: Supports jpg, jpeg, png, bmp, tiff, webp formats
3. **Memory Usage**: Pay attention to memory usage when processing large images or batches
4. **Output Overwrite**: Files with the same name will be overwritten

## 🐛 FAQ

**Q: Unable to load model?**  
A: Ensure `mobile_sam.pt` is in the correct path. You can use the `-m` parameter to specify the path.

**Q: Generated mask is inaccurate?**  
A: Try:
- Interactive Batch: Draw multiple boxes or adjust box positions
- Auto Batch: Switch to different processing modes
- Single Image Interactive: Use point+box mixed mode

**Q: Processing speed is slow?**  
A: 
- Use GPU: Install CUDA version of PyTorch
- Reduce image size
- Use simple configuration in auto batch mode

## 📝 License

This project is based on MobileSAM and Ultralytics. Please comply with the corresponding open source licenses.

## 🙏 Acknowledgments

- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [Ultralytics](https://github.com/ultralytics/ultralytics)

## 🌟 Star History

If this project helps you, welcome to give it a Star ⭐️!

## 📧 Contact

For questions or suggestions, feel free to submit an Issue or Pull Request.

---

<div align="center">

**Good-Segment** - Make Image Segmentation Simpler ✨

Made with ❤️ by [Spike Don]

</div>
