# Generative AI Art

This repository explores various generative adversarial networks (GANs) and other generative models for creating unique and artistic images. It includes implementations of different GAN architectures and training methodologies.

## Features

- **DCGAN Implementation**: A deep convolutional GAN for generating images.
- **Training Utilities**: Scripts for training GANs on datasets like MNIST.
- **Image Generation**: Functions to generate new images from trained models.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training a GAN Example

```python
import tensorflow as tf
import numpy as np
from gan_model import GAN

# Load MNIST dataset
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Rescale -1 to 1
X_train = X_train / 127.5 - 1.
X_train = np.expand_dims(X_train, axis=3)

gan = GAN()
gan.train(X_train, epochs=100, batch_size=32, sample_interval=10)
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.
