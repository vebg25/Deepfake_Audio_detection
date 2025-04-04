import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

class AudioEvaluation:
    def evaluate(self, X_test, y_test):
        """Evaluate the model on the test dataset."""
        # Make predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Calculate metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        # Calculate Equal Error Rate (EER)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        fnr = 1 - tpr

        # Find the threshold where FPR and FNR are equal (EER)
        eer_threshold_idx = np.nanargmin(np.absolute(fpr - fnr))
        eer = np.mean([fpr[eer_threshold_idx], fnr[eer_threshold_idx]])

        # Print results
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)
        print(f"\nEqual Error Rate (EER): {eer:.4f}")

        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.scatter(fpr[eer_threshold_idx], fnr[eer_threshold_idx], marker='o', color='red',
                   label=f'EER = {eer:.4f}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)

        return {
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'eer': eer,
            'auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }