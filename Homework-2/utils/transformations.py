import random


def augment_dataset_with_replacement(dataset, target_size, augmentation_transforms):
    """
    Augment images until a desired set size is reached. Images are picked randomly with replacement.
    """
    augmented_dataset = []
    dataset_size = len(dataset)

    while len(augmented_dataset) < target_size:
        idx = random.randint(0, dataset_size - 1)
        example = dataset[idx]
        augmented_image = augmentation_transforms(example["image"])
        augmented_example = {
            "text": example["text"],
            "image": augmented_image,
        }
        augmented_dataset.append(augmented_example)

    return augmented_dataset


def resize_dataset(dataset, resize_transform):
    resized_dataset = []
    for example in dataset:
        resized_image = resize_transform(example["image"])
        resized_example = {
            "text": example["text"],
            "image": resized_image,
        }
        resized_dataset.append(resized_example)
    return resized_dataset
