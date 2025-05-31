import random
import matplotlib.pyplot as plt
from cnn_model import CNN
from loss import CrossEntropyLoss
from optimizer import GradientDescent
from utils import load_mnist_data

def plot_live_progress(epoch_numbers, train_losses, test_accuracies, epoch):
    """Live plot of training progress (updates every 5 epochs)"""
    if epoch % 5 == 0:
        plt.clf()

        plt.subplot(1, 2, 1)
        plt.plot(epoch_numbers, train_losses, 'b-', linewidth=2, marker='o', markersize=4)
        plt.title(f'Training Loss (Epoch {epoch})')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(epoch_numbers, test_accuracies, 'g-', linewidth=2, marker='s', markersize=4)
        plt.title(f'Test Accuracy (Epoch {epoch})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.pause(0.01)


def plot_sample_predictions(model, test_dataset, num_samples=6):
    """Visualize sample predictions from test set"""
    plt.figure(figsize=(12, 8))
    samples = random.sample(test_dataset, num_samples)

    for i, (image, true_label) in enumerate(samples):
        plt.subplot(2, 3, i + 1)

        # Reshape for grayscale
        img_2d = image[0] if len(image) == 1 and len(image[0]) == 28 else image
        plt.imshow(img_2d, cmap='gray')

        preds = model.forward(image)
        pred_digit = preds.index(max(preds))
        actual_digit = true_label.index(1)
        confidence = max(preds)

        color = 'green' if pred_digit == actual_digit else 'red'
        plt.title(f'Pred: {pred_digit}, True: {actual_digit}\nConf: {confidence:.2f}',
                  color=color, fontweight='bold')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()


# model, loss, optimizer setup
model = CNN(config_path="config.txt")
loss_function = CrossEntropyLoss()
optimizer = GradientDescent(model.layers, learning_rate=0.03)

# load a reduced MNIST dataset
train_dataset = load_mnist_data(num_samples=1000, kind='train')
test_dataset = load_mnist_data(num_samples=200, kind='test')

num_train = len(train_dataset)
num_test = len(test_dataset)

# hyperparameters
batch_size = 8
epochs = 5

train_losses = []
test_accuracies = []
epoch_numbers = []

print(f"Training on {num_train} samples | Testing on {num_test} samples | Epochs={epochs} | Batch size={batch_size} | LR=0.03\n")

for epoch in range(epochs):
    total_loss = 0.0
    random.shuffle(train_dataset)

    for start in range(0, num_train, batch_size):
        batch = train_dataset[start: start + batch_size]

        for image, true_label in batch:
            preds = model.forward(image)
            loss = loss_function.forward(preds, true_label)
            total_loss += loss

            grad_from_loss = loss_function.backward()
            param_grads = model.backward(grad_from_loss)
            optimizer.step(param_grads)

        print(f"Epoch {epoch+1}/{epochs} | Batch {start//batch_size+1}/{(num_train+batch_size-1)//batch_size} "
              f"| Loss so far: {total_loss / (start + len(batch)):.4f}", end='\r')

    avg_loss = total_loss / num_train
    correct = 0
    for image, true_label in test_dataset:
        preds = model.forward(image)
        pred_digit = preds.index(max(preds))
        actual_digit = true_label.index(1)
        if pred_digit == actual_digit:
            correct += 1
    acc = correct / num_test

    train_losses.append(avg_loss)
    test_accuracies.append(acc)
    epoch_numbers.append(epoch + 1)

    plot_live_progress(epoch_numbers, train_losses, test_accuracies, epoch + 1)

    print(f"\nEpoch {epoch+1}/{epochs} -> Avg Loss: {avg_loss:.4f} | "
          f"Test Acc: {acc:.2%} ({correct}/{num_test})\n")

print("Training complete.")

# result visualization
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(epoch_numbers, train_losses, 'b-', linewidth=2, marker='o', markersize=4)
plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.subplot(1, 3, 2)
plt.plot(epoch_numbers, test_accuracies, 'g-', linewidth=2, marker='s', markersize=4)
plt.title('Test Accuracy Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.tight_layout()

plt.subplot(1, 3, 3)
ax1 = plt.gca()
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss', color='tab:red')
ax1.plot(epoch_numbers, train_losses, color='tab:red', linewidth=2, marker='o', markersize=4)
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.set_ylabel('Test Accuracy', color='tab:blue')
ax2.plot(epoch_numbers, test_accuracies, color='tab:blue', linewidth=2, marker='s', markersize=4)
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.set_ylim(0, 1)

plt.title('Training Progress', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Final Training Loss: {train_losses[-1]:.4f}")
print(f"Final Test Accuracy: {test_accuracies[-1]:.2%}")
print(f"Best Test Accuracy: {max(test_accuracies):.2%} (Epoch {test_accuracies.index(max(test_accuracies)) + 1})")
print(f"Lowest Training Loss: {min(train_losses):.4f} (Epoch {train_losses.index(min(train_losses)) + 1})")

print("\nGenerating sample predictions...")
plot_sample_predictions(model, test_dataset, num_samples=6)
