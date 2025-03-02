import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def create_mask(shape, ratio):
    return np.random.rand(*shape[:2]) < ratio


def apply_mask(image, mask):
    return image * mask[:, :, None]


def shepard_interpolation(image, mask, window=9, p=2):
    h, w, _ = image.shape
    output = np.copy(image)

    y, x = np.where(mask == 0)

    for i, j in zip(y, x):
        i_min, i_max = max(0, i - window // 2), min(h, i + window // 2 + 1)
        j_min, j_max = max(0, j - window // 2), min(w, j + window // 2 + 1)

        local_pixels = image[i_min:i_max, j_min:j_max]
        local_mask = mask[i_min:i_max, j_min:j_max]

        valid_coords = np.argwhere(local_mask)
        valid_pixels = local_pixels[local_mask.astype(bool)]

        if valid_coords.size == 0:
            continue

        distances = np.linalg.norm(valid_coords + [i_min, j_min] - [i, j], axis=1) ** p
        weights = 1.0 / np.maximum(distances, 1e-6)
        weights /= weights.sum()

        output[i, j] = np.dot(weights, valid_pixels)

    return np.clip(output, 0, 1)


def show_images(images, titles):
    plt.figure(figsize=(12, 6))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    image_path = "src/utils/test.bmp"
    image = load_image(image_path)
    mask = create_mask(image.shape, 0.2)

    masked_image = apply_mask(image, mask)
    interpolated_image = shepard_interpolation(masked_image, mask)

    show_images(
        [masked_image, interpolated_image], ["Masked Image", "Interpolated Image"]
    )


if __name__ == "__main__":
    main()
