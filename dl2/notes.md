# Notes

## Objective

> The goal of __Deep Learning from the Foundations__ course is to deepen my practice in understanding and building cutting edge deep learning and make them work and perform at an acceptable speed.

The main objective of this course is to learn deep learning from the foundations by building everything from scratch. This will allow us to implement papers and enhance our engineering skills because the best practitioners are the ones that can put ideas into code and make it work. Since the field is evolving rapidly and whatever cutting edge architecture today might be useless in a short period of time, the ability to read research papers and implement new ideas/architecture will be the most valuable skill for any DL practitioner.

## Swift Language

Swift has a lot of potential in the ML and DL world. It's very fast and considered a numerical computing language. One of the main advantages of Swift is that there is not any other language between the compiler and Swift and is considered a thin layer that sits on top of the compiler. This means that you can debug the code to the lower level and see what the computer is doing as opposed to PyTorch where the code gets translated to C++/C before getting compiled. The last two lessons would be covering Swift in greater detail.

## WHY

When you create it, you understand it better. This would allow you to tweak everything and know how everything works under the hood. As a result, you can contribute to open-source libraries and correlate papers with code.

## Steps to train any model

1. Build the model that overfits the data. Overfitting means that the validation error starts increasing (change direction from decreasing). Overfitting doesn't mean that training error is less than validation error because almost always training error is less than validation error.
2. Reduce overfitting by:
    1. More data.
    2. Data augmentation.
    3. More generalizable architectures.
    4. Regularization.
    5. Reduce architecture complexity.
3. There is no step 3. Basically visualizing model's output and make sure everything looks reasonable.  
