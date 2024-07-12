# MNIST Digit Classifier: Detecting the Digit '2' Using Logistic Regression

## Overview
This project demonstrates the process of building a machine learning model to classify handwritten digits from the MNIST dataset, focusing specifically on detecting the digit '2'. The model uses Logistic Regression and achieves its evaluation through cross-validation.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset
The MNIST dataset is a large collection of handwritten digits that is commonly used for training various image processing systems. It contains 70,000 grayscale images of 28x28 pixels each.

## Project Structure
```
MNIST-Digit-Classifier/
│
├── mnist_classifier.py        # Main code file for the project
├── README.md                  # Readme file for the project
├── requirements.txt           # Python dependencies
└── results.txt                # File to store results
```

## Installation
1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/MNIST-Digit-Classifier.git
    cd MNIST-Digit-Classifier
    ```

2. **Create a Virtual Environment:**
    ```bash
    python -m venv env
    source env/bin/activate   # On Windows: env\Scripts\activate
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Run the Script:**
    ```bash
    python mnist_classifier.py
    ```

2. **View Results:**
    The output will include a visualization of a sample digit, the prediction for the sample digit, and the cross-validation accuracy scores.

## Results
The model predicts whether a given image is the digit '2' with the following cross-validation accuracy scores:
```
Cross-validation accuracy scores: [0.98, 0.97, 0.98]
Mean cross-validation accuracy: 0.9767
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or additional features.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
