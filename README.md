# WiFi CSI Indoor Localization with Deep Learning

Implementation of "Hybrid CNN-LSTM based Robust Indoor Pedestrian Localization with CSI Fingerprint Maps" using TensorFlow. This system achieves fine-grained indoor positioning using WiFi Channel State Information (CSI) and deep learning.

## Overview

This repository provides a TensorFlow implementation of a novel indoor positioning technique that combines LSTM neural networks with CSI fingerprinting. The system processes WiFi CSI data to achieve sub-meter localization accuracy without requiring additional sensors.

## Key Features

- LSTM-based deep learning architecture for CSI processing
- Multi-level building and floor classification 
- Batch processing support for efficient training
- CSI data preprocessing and normalization
- Support for continuous location tracking

## Requirements

```
tensorflow>=1.x
numpy
pandas
scipy
```

## Installation

```bash
git clone https://github.com/username/wifi-csi-localization.git
cd wifi-csi-localization
pip install -r requirements.txt
```

## Usage

1. Prepare your CSI data in CSV format with the following columns:
   - CSI measurements (520 features)
   - BUILDINGID
   - FLOOR

2. Train the model:
```python
python multilab_lstm_loc.py
```

The script will:
- Load and preprocess CSI data
- Split data into training/test sets (80/20)
- Train the LSTM model  
- Output training accuracy metrics

## Model Architecture

The system uses a stacked LSTM architecture with:
- Input dimension: 520 CSI features
- 2 LSTM layers
- Hidden units: 520
- Batch size: 10
- Learning rate: 0.001
- Training epochs: 100

## Data Format

Training data should be in CSV format with:
- First 520 columns: CSI measurements
- BUILDINGID: Building identifier
- FLOOR: Floor number

## Performance

The model achieves:
- Sub-meter accuracy in static environments
- Robust performance across different buildings and floors
- Real-time prediction capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Background Research

See link to the base research article.
```
http://www.researchgate.net/publication/387083505_Hybrid_CNN-LSTM_based_Robust_Indoor_Pedestrian_Localization_with_CSI_Fingerprint_Maps#fullTextFileContent 
```

## License

MIT License

## Acknowledgments

This implementation is based on research conducted at Texas A&M University's Department of Computer Science and Engineering.

## Contact

For questions and support, please open an issue in the GitHub repository.

---

**Note**: This is a research implementation. For production use, please refer to the full paper for additional optimizations and considerations.
