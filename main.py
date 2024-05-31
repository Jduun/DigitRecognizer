import tkinter as tk
import numpy as np
import torch
import nn_model

# I make a 280x280 grid consisting of only 0 and 255,
# then I compress it to 28x28
# (this is the right size for a neural network),
# averaging the squares by 10x10,
# thereby getting the image closest to MNIST.
# This is not displayed on the canvas,
# these changes are only in the array


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DigitRec")

        self.canvas_size = 280
        self.cell_size = 1
        self.grid_size = 280
        self.pen_color = "black"

        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        self.grid = np.zeros((self.grid_size, self.grid_size))

        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.save_button = tk.Button(self.root, text="Predict", command=self.predict_digit)
        self.save_button.pack()

        self.prediction_label = tk.Label(self.root, text="Predicted Label: None")
        self.prediction_label.pack()

    def paint(self, event):
        x, y = event.x, event.y
        cell_x, cell_y = x // self.cell_size, y // self.cell_size

        if 0 <= cell_x < self.grid_size and 0 <= cell_y < self.grid_size:
            self.canvas.create_rectangle(
                cell_x * self.cell_size, cell_y * self.cell_size,
                (cell_x + 20) * self.cell_size, (cell_y + 20) * self.cell_size,
                fill=self.pen_color, outline=self.pen_color
            )
            # This is done to bring the drawn image closer to MNIST
            for y in range(cell_y, cell_y + 20):
                for x in range(cell_x, cell_x + 20):
                    self.grid[y, x] = 255
            for y in range(cell_y, cell_y + 17):
                for x in range(cell_x, cell_x + 17):
                    self.grid[y, x] = 160

    def clear_canvas(self):
        self.canvas.delete("all")
        self.grid = np.zeros((self.grid_size, self.grid_size))

    def predict_digit(self):
        scaled_array = self.grid.reshape((28, 10, 28, 10)).mean(axis=(1, 3)).astype(np.uint8)
        print(scaled_array)
        # Adding the size of the batch and channel
        tensor_image = torch.tensor(scaled_array, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        # Make prediction
        with torch.no_grad():
            # Passing the tensor through a neural network
            y_pred = model(tensor_image.to(device))
            # Find the index of the maximum logit to get the predicted label
            _, predicted_label = torch.max(y_pred, 1)
        self.prediction_label.config(text=f"Predicted Label: {predicted_label.item()}")


if __name__ == "__main__":
    global model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = nn_model.SimpleConvNet().to(device)
    model.load_state_dict(torch.load('nn_model_weights.pth'))

    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()

