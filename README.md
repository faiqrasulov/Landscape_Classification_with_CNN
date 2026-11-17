# Landscape image classification with a CNN (a failed run, diagnosed)

> Course assignment from 2023, kept as it was written. The notes under "Reading the result" were added in November 2026.

A PyTorch CNN for the Intel image classification data: about 14,000 training and 3,000 test
photos, 150 x 150 px, in six classes (buildings, forest, glacier, mountain, sea, street).

## Result

**The model did not learn.** Test accuracy is 14.5%; guessing gives 16.7%. The training loss
prints 6.2383 at every step of all six epochs.

## Why

The bug is in the last line of `forward`:

```python
X = F.relu(self.fc2(X))   # 512 features
out = self.out(X)         # class scores
return X                  # returns the features, not the scores
```

The loss was computed on the 512 hidden features instead of the class scores. 6.2383 is ln(512):
the loss of a uniform guess over 512 outputs, which is what 512 equal features give. The output
layer never entered the loss, so it was never trained.

Three more things to fix once `return out` is in place:

- the output layer has 10 units for 6 classes;
- the two convolutions have no activation between them, and the images are not normalised;
- the test images go through `RandomResizedCrop`, so the test score would vary between runs.

The notebook is kept unchanged as a record. The check that would have caught it: the first
printed loss of an untrained 6-class model should be near ln(6) = 1.79.

## Newer work on the same topic

- [autograd-from-scratch](https://github.com/faiqrasulov/autograd-from-scratch): a CNN whose every gradient is checked against PyTorch.

## Files

| File | Content |
|---|---|
| `Landscape_Classification_with_CNN.ipynb` | Notebook (written for Google Colab) |
| `Landscape_Classification_with_CNN.html` | Rendered copy |
| `About.txt` | Original brief |

The images are not in the repository.
