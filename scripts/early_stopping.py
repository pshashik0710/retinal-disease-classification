import torch


# =========================================================
# EARLY STOPPING
# =========================================================

class EarlyStopping:

    def __init__(
        self,
        patience=7,
        mode="max",
        delta=0.0,
        checkpoint_path="best_model.pth"
    ):

        self.patience = patience

        self.mode = mode

        self.delta = delta

        self.checkpoint_path = checkpoint_path

        self.counter = 0

        self.best_score = None

        self.early_stop = False

    # =====================================================
    # CHECK IMPROVEMENT
    # =====================================================

    def __call__(
        self,
        score,
        model
    ):

        # ---------------------------------------------
        # FIRST SCORE
        # ---------------------------------------------

        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(model)

            print(f"\nInitial best score: {score:.4f}")

            return

        # ---------------------------------------------
        # MAX MODE
        # Example:
        # Accuracy
        # F1
        # ROC-AUC
        # ---------------------------------------------

        if self.mode == "max":

            improved = score > (
                self.best_score + self.delta
            )

        # ---------------------------------------------
        # MIN MODE
        # Example:
        # Validation Loss
        # ---------------------------------------------

        else:

            improved = score < (
                self.best_score - self.delta
            )

        # ---------------------------------------------
        # IF IMPROVED
        # ---------------------------------------------

        if improved:

            self.best_score = score

            self.counter = 0

            self.save_checkpoint(model)

            print(
                f"\nValidation improved "
                f"-> Best Score: {score:.4f}"
            )

        # ---------------------------------------------
        # NO IMPROVEMENT
        # ---------------------------------------------

        else:

            self.counter += 1

            print(
                f"\nNo improvement "
                f"{self.counter}/{self.patience}"
            )

            if self.counter >= self.patience:

                self.early_stop = True

                print("\nEarly stopping triggered.")

    # =====================================================
    # SAVE CHECKPOINT
    # =====================================================

    def save_checkpoint(
        self,
        model
    ):

        torch.save(
            model.state_dict(),
            self.checkpoint_path
        )