# Car Make & Model Recognition

A deep learning project for **car make and model classification** using **PyTorch** on the **NVIDIA Jetson Orin Nano**. This project classifies vehicles from images into their corresponding make, model, and year using the Stanford Cars dataset.

## Features

* 🚗 Classifies **196 different car make/model/year classes**
* 🖼️ Predicts from uploaded or randomly selected images
* ⚡ GPU-accelerated inference on NVIDIA Jetson
* 🌐 Simple web interface for testing predictions
* 🔀 Random image demo for quick testing

---

## Hardware

* NVIDIA Jetson Orin Nano Developer Kit
* CUDA-enabled GPU
* JetPack 6.x
* Python 3.10+

---

## Dataset

This project was trained using the **Stanford Cars Dataset by Classes Folder** available on Kaggle.

Download it here:

**https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder/data**

The dataset contains:

* **16,185 images**
* **196 unique car classes**
* **8,144 training images**
* **8,041 testing images**

Each class represents a unique combination of:

* Make
* Model
* Year

For example:

* 2012 Tesla Model S
* 2012 BMW M3 Coupe
* 2012 Ford Edge SUV

The Kaggle version already organizes the images into folders, making it easy to use with PyTorch's `ImageFolder` dataset.

---

## Dataset Structure

After downloading, organize the dataset like this:

```text
dataset/
├── train/
│   ├── Acura Integra Type R 2001/
│   ├── Audi S4 Sedan 2007/
│   ├── ...
│
└── test/
    ├── Acura Integra Type R 2001/
    ├── Audi S4 Sedan 2007/
    ├── ...
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/kyneh13/car-make-model.git
cd car-make-model
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Training

Place the downloaded dataset inside the project folder.

Example:

```text
dataset/
    train/
    test/
```

Then start training:

```bash
python train.py
```

Training will automatically learn all available classes contained in the dataset.

---

## Running Inference

Predict a single image:

```bash
python predict.py path/to/image.jpg
```

Example output:

```text
1. Ford Edge SUV 2012 - 98.7%
2. Honda CR-V 2012 - 0.9%
3. Jeep Grand Cherokee 2012 - 0.4%
```

---

## Random Demo

Run the random image demonstration:

```bash
python random_demo.py
```

The program will:

* Select a random test image
* Display the image
* Predict the vehicle
* Show the top predictions with confidence scores

---

## Web Demo

Launch the web interface:

```bash
python random_demo_web.py
```

Then open your browser and navigate to:

```
http://localhost:8000
```

The web demo allows you to:

* View a random test image
* Run predictions
* Display confidence scores

---

## Project Structure

```text
car-make-model/
│
├── dataset/
├── models/
├── test_images/
├── train.py
├── predict.py
├── random_demo.py
├── random_demo_web.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

* Python
* PyTorch
* TorchVision
* CUDA
* NVIDIA Jetson
* HTML
* JavaScript

---

## Future Improvements

* Live webcam detection
* Vehicle price estimation
* Model year trend analysis
* Support for custom-trained datasets
* Mobile-friendly web interface

---

## Acknowledgements

This project uses the **Stanford Cars Dataset** created by Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. The folder-organized Kaggle version used for training is maintained by Jesús Utrera.

---

## License

This project is released under the MIT License.

Please refer to the original dataset license on Kaggle before redistributing the dataset.
