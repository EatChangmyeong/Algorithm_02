# Handwritten Digit Recognition with TensorFlow

*This repository is a part of Algorithm 02 Assignment; the original repository is [Jin0316/Algorithm](https://github.com/Jin0316/Algorithm).*

You can check out the (edited) notebook at [/TensorFlow_mnist_example.ipynb](https://github.com/EatChangmyeong/Algorithm_02/blob/main/TensorFlow_mnist_example.ipynb); use [nbviewer](https://nbviewer.jupyter.org/github/EatChangmyeong/Algorithm_02/blob/main/TensorFlow_mnist_example.ipynb) if GitHub viewer does not work.

## The Models

I used the three models provided as examples as-is, stored as `model1`, `model2`, and `model3` respectively.

![Model definition and declaration.](https://user-images.githubusercontent.com/25813580/121696183-c4fba300-cb06-11eb-9a50-d8194d83ae26.png)

```python
def select_model(model_number):
    if model_number == 1:
        model = keras.models.Sequential([
                    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),    # layer 1 
                    keras.layers.MaxPool2D((2, 2)),                                                 # layer 2 
                    keras.layers.Flatten(),
                    keras.layers.Dense(10, activation='softmax')])                                  # layer 3

    if model_number == 2:
        model = keras.models.Sequential([
                    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),    # layer 1 
                    keras.layers.MaxPool2D((2, 2)),                                                 # layer 2
                    keras.layers.Conv2D(64, (3, 3), activation='relu'),                             # layer 3 
                    keras.layers.MaxPool2D((2, 2)),                                                 # layer 4
                    keras.layers.Flatten(),
                    keras.layers.Dense(10, activation='softmax')])                                  # layer 5
                    
    if model_number == 3: 
        model = keras.models.Sequential([
                    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),    # layer 1
                    keras.layers.MaxPool2D((2, 2)),                                                 # layer 2
                    keras.layers.Conv2D(64, (3, 3), activation='relu'),                             # layer 3
                    keras.layers.Conv2D(64, (3, 3), activation='relu'),                             # layer 4
                    keras.layers.MaxPool2D((2, 2)),                                                 # layer 5
                    keras.layers.Conv2D(128, (3, 3), activation='relu'),                            # layer 6
                    keras.layers.Flatten(),
                    keras.layers.Dense(10, activation='softmax')])                                  # layer 7
    
    return model

model1 = select_model(1)
model2 = select_model(2)
model3 = select_model(3)
```

The `summary()` call reveals that each model has 54,090, 34,826, and 134,730 trainable parameters respectively; I expected that model with more trainable parameters would be more accurate in general.

## Training

Each model was tranied with the same training images and labels from the MNIST dataset with 5 epochs.

![Training results.](https://user-images.githubusercontent.com/25813580/121697054-a4801880-cb07-11eb-99d0-9d68172c4f4d.png)

After the last epoch,
* Model 1's accuracy and loss was 0.9834 and 0.0568.
* Model 2's accuracy and loss was 0.9868 and 0.0429.
* Model 3's accuracy and loss was 0.9907 and 0.0313.

## Test Result

Each model was then tested with the same testing images and labels from the MNIST dataset.

![Test results.](https://user-images.githubusercontent.com/25813580/121697955-836bf780-cb08-11eb-87e8-a76df84e242e.png)

* Model 2 was the most accurate one, with accuracy 0.9857 and loss 0.0517.
* Model 3 follows with accuracy 0.9792 and loss 0.0657.
* Model 1 was the least accurate, with accuracy 0.9682 and loss 0.1430.

This result disproves my former hypothesis where more trainable parameters meant more accuracy.

## Images That Were Correctly Predicted

Each model predicted the first fifteen images similarly, with almost perfect confidence for each image.

![Model 1's predictions.](https://user-images.githubusercontent.com/25813580/121698871-6552c700-cb09-11eb-9e34-efc4675d4a9e.png)
![Model 2's predictions.](https://user-images.githubusercontent.com/25813580/121699010-85828600-cb09-11eb-9898-6a33de4a1d6f.png)
![Model 3's predictions.](https://user-images.githubusercontent.com/25813580/121699075-94693880-cb09-11eb-8abe-fc3345ae14d1.png)

## Images That Were Incorrectly Predicted

Each model produced 318, 143, and 208 mispredictions; the first ten examples for each follows.

![Ten incorrect predictions for each model.](https://user-images.githubusercontent.com/25813580/121699385-d85c3d80-cb09-11eb-8956-6f2dbb0e2605.png)
