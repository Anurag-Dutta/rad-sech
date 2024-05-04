# Radicalized Family of Hyperbolic Secant (`RadSech`) as an activation function

_Abstract_

Neural Networks have proved their capabilities to model several instances of the modern day. From the last few decades, Machine Intelligence caught great attention, but from the starting of this decade, it rose to great prominence, with the involvement of Artificial Intelligence in a broad (almost every) spectrum of STEM (Science, Technology, Engineering and Management). However, complex the tasks that are being performed by the Neural Networks, the basis behind these architecture to map data points, that are difficult to be separated linearly, to a linearly separable cluster using a combination of neural layers, which are simply a span of linear, (or non - linear) functions, that are better known as Activation Functions. Amongst most commonly used Activation Functions are Step Functions, Sigmoid, Hyperbolic Tangent, and Rectified Linear Unit (`ReLU`). Herein, a novel class of Radicalized Hyperbolic Secant Activation Functions (`RadSech`), is proposed i.e., ${\rm sech}^{\frac{1}{n}}\left(\cdot\right)$, for $n\in\mathbb{Z}^+$. The efficacy of the proposed Function has been demonstrated by its’ abilities to outperform its’ counterparts, like the Gaussian Activation Function, and Gaussian Error Linear Unit (`GeLU`). Besides being simpler in action than the counterparts, the robustness of `RadSech` have been proven on various datasets offered by the `scikit-learn` library of Python, and across different model complexities.


<img src="https://github.com/Anurag-Dutta/rad-sech/blob/main/rad_sech.jpg"  alt="RadSecH">
  <figcaption> <p align="center"> Fig. RadSecH </p>  </figcaption>


```python
import torch

def rad_sech(x: torch.Tensor, n: int) -> torch.Tensor:
    return (2 / (torch.exp(x) + torch.exp(-x)))**(1/n)

# Example usage
input_tensor = torch.tensor([1.0, 2.0, 3.0])
output_tensor = rad_sech(input_tensor, 2)
print(output_tensor)
```




<img src="https://github.com/Anurag-Dutta/rad-sech/blob/main/scaled_rad_sech.jpg"  alt="RadSecH">
  <figcaption> <p align="center"> Fig. Scaled RadSecH </p>  </figcaption>


```python
import torch

def scaled_rad_sech(x: torch.Tensor, n: int) -> torch.Tensor:
    return x * (2 / (torch.exp(x) + torch.exp(-x)))**(1/n)

# Example usage
input_tensor = torch.tensor([1.0, 2.0, 3.0])
output_tensor = scaled_rad_sech(input_tensor, 2)
print(output_tensor)
```
