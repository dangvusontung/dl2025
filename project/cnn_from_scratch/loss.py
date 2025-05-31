# loss.py
import math

class CrossEntropyLoss:
    def __init__(self):
        # Dùng để lưu lại giá trị cho pha backward
        self.predicted_probs = None
        self.true_labels = None

    def forward(self, predicted_probs, true_labels):
        """
        Tính Cross-Entropy loss.
        
        Args:
            predicted_probs (list): Vector xác suất từ output của mô hình (sau softmax).
            true_labels (list): Vector one-hot encoding của nhãn đúng. Ví dụ: [0, 0, 1, 0, ...]
        
        Returns:
            float: Giá trị loss.
        """
        self.predicted_probs = predicted_probs
        self.true_labels = true_labels
        
        loss = 0.0
        # Thêm một số rất nhỏ (epsilon) để tránh tính log(0)
        epsilon = 1e-9
        
        for i in range(len(predicted_probs)):
            loss -= true_labels[i] * math.log(predicted_probs[i] + epsilon)
            
        return loss

    def backward(self):
        """
        Tính gradient của loss theo output của softmax.
        Đây là gradient ban đầu để bắt đầu quá trình backpropagation.
        Công thức: dL/d_out = predicted_probs - true_labels
        """
        gradient = []
        for i in range(len(self.predicted_probs)):
            gradient.append(self.predicted_probs[i] - self.true_labels[i])
            
        return gradient