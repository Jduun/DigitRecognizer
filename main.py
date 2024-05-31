import tkinter as tk
import numpy as np
import torch


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")

        self.canvas_size = 280
        self.cell_size = 10
        self.grid_size = 28
        self.pen_color = "black"

        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        self.grid = np.zeros((self.grid_size, self.grid_size))

        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.save_button = tk.Button(self.root, text="Save", command=self.save_canvas)
        self.save_button.pack()

    def paint(self, event):
        x, y = event.x, event.y
        cell_x, cell_y = x // self.cell_size, y // self.cell_size

        if 0 <= cell_x < self.grid_size and 0 <= cell_y < self.grid_size:
            self.canvas.create_rectangle(
                cell_x * self.cell_size, cell_y * self.cell_size,
                (cell_x + 1) * self.cell_size, (cell_y + 1) * self.cell_size,
                fill=self.pen_color, outline=self.pen_color
            )
            self.grid[cell_y, cell_x] = 1

    def clear_canvas(self):
        self.canvas.delete("all")
        self.grid = np.zeros((self.grid_size, self.grid_size))

    def save_canvas(self):
        np.savetxt("digit.csv", self.grid, delimiter=",", fmt='%d')
        print("Canvas saved to digit.csv")


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
