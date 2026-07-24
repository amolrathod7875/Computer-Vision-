# CV_Assignment_2 - Image Affine Transformations

## Theory

Affine transformations are a class of geometric transformations that preserve:
- **Collinearity**: Points on a line remain on a line after transformation
- **Parallelism**: Parallel lines remain parallel
- **Ratios of distances**: Points dividing a line segment keep the same ratio

An affine transformation in 2D is represented using **homogeneous coordinates** (3×3 matrix) or **affine matrix** (2×3 matrix in OpenCV). Any affine transformation can be decomposed into a combination of translation, rotation, scaling, and shearing.

## Mathematical Formulas

### 1. Translation
Shifts every point by a fixed offset:
```
x' = x + tx
y' = y + ty
```
**Matrix form (2×3):**
```
T = [ [1, 0, tx],
      [0, 1, ty] ]
```

### 2. Rotation
Rotates points around the origin by angle θ:
```
x' = x·cosθ - y·sinθ
y' = x·sinθ + y·cosθ
```
**Matrix form (2×2):**
```
R(θ) = [ [cosθ, -sinθ],
         [sinθ,  cosθ] ]
```

### 3. Scaling
Scales points along x and y axes:
```
x' = sx · x
y' = sy · y
```
**Matrix form (2×2):**
```
S = [ [sx, 0],
      [0, sy] ]
```

### 4. Combined Transformation
Applying scale → rotation → translation sequentially:
```
M_total = T(tx, ty) · R(θ) · S(sx, sy)
```
Where matrix multiplication is non-commutative, so order matters.

## Mermaid Flow Diagram

```mermaid
flowchart TD
    A[Start] --> B{Load Image?}
    B -->|Success| C[Display Original Image]
    B -->|Failure| D[Use Synthetic Image]
    D --> C
    
    C --> E[Translation<br/>M = [[1,0,tx],[0,1,ty]]]
    E --> F[Display Translated]
    
    F --> G[Rotation<br/>M = cv2.getRotationMatrix2D]
    G --> H[Display Rotated]
    
    H --> I[Scaling<br/>M = [[sx,0,(1-sx)·cx],[0,sy,(1-sy)·cy]]]
    I --> J[Display Scaled]
    
    J --> K[Combined Transform<br/>M_total = T·R·S]
    K --> L[Display Combined Result]
    
    L --> M[End]
```

## Important Questions and Answers

### Q1: What is an affine transformation?
An affine transformation is a linear mapping that preserves collinearity and ratios of distances. It combines translation, rotation, scaling, and shearing into a single matrix operation.

### Q2: Why do we use homogeneous coordinates?
Homogeneous coordinates allow us to represent translation as matrix multiplication (not addition), enabling all affine operations to be expressed as a single matrix product.

### Q3: What is the difference between a 2×2 and 2×3 affine matrix in OpenCV?
OpenCV uses 2×3 affine matrices for efficiency. The first two columns represent the linear transformation (rotation/scale/shear), and the third column represents translation.

### Q4: Why is matrix multiplication order important?
Matrix multiplication is non-commutative. The order determines which transformation is applied first. For example, `T·R` means rotate first, then translate.

### Q5: What happens if we scale about the origin instead of the image center?
Scaling about the origin moves the image away from its center, potentially causing parts of the image to move out of the frame. Scaling about the center keeps the image centered.

### Q6: How does OpenCV's `warpAffine` handle out-of-bounds pixels?
By default, `warpAffine` leaves out-of-bounds areas black (0). This can be changed using `borderMode` and `borderValue` parameters.

### Q7: What is the purpose of `cv2.BORDER_CONSTANT`?
It fills pixels outside the transformed image boundaries with a constant color specified by `borderValue`, preventing artifacts at the edges.

### Q8: Why use `np.float32` for transformation matrices?
OpenCV expects floating-point matrices for affine transformations to support sub-pixel interpolation and accurate geometric computations.

### Q9: What is the role of interpolation in `warpAffine`?
Interpolation determines how pixel values are computed when source coordinates are non-integer. Default is linear interpolation (`cv2.INTER_LINEAR`).

### Q10: Can affine transformations warp a rectangle into a general quadrilateral?
No. Affine transformations preserve parallelism, so a rectangle remains a parallelogram. To warp into an arbitrary quadrilateral, you need a perspective (projective) transformation using `cv2.getPerspectiveTransform`.
