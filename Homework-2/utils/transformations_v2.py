import random
from collections import defaultdict

def augment_dataset_with_replacement(dataset, target_size, augmentation_transforms):
    """
    Augment images class-wise until each class reaches the desired size.
    """
    class_counts = defaultdict(int)
    for example in dataset:
        class_counts[example["class"]] += 1

    total_classes = len(class_counts)
    target_per_class = target_size // total_classes

    augmented_dataset = []
    for class_label, count in class_counts.items():
        # Calculate how many samples are needed for this class
        augment_count = max(target_per_class - count, 0)

        # Get indices of all images in this class
        class_indices = [i for i, example in enumerate(dataset) if example["class"] == class_label]

        # Augment images for this class
        while augment_count > 0:
            idx = random.choice(class_indices)
            example = dataset[idx]
            augmented_image = augmentation_transforms(example["image"])
            augmented_example = {
                "image": augmented_image,
                "class": example["class"],
            }
            augmented_dataset.append(augmented_example)
            augment_count -= 1

    # Add original dataset images to the augmented dataset
    augmented_dataset.extend(dataset)

    # If the dataset is still smaller than target_size, randomly add more augmented images
    while len(augmented_dataset) < target_size:
        idx = random.randint(0, len(dataset) - 1)
        example = dataset[idx]
        augmented_image = augmentation_transforms(example["image"])
        augmented_example = {
            "image": augmented_image,
            "class": example["class"],
        }
        augmented_dataset.append(augmented_example)

    return augmented_dataset


def resize_dataset(dataset, resize_transform):
    resized_dataset = []
    for example in dataset:
        resized_image = resize_transform(example["image"])
        resized_example = {
            "image": resized_image,
            "class": example["class"],
        }
        resized_dataset.append(resized_example)
    return resized_dataset
