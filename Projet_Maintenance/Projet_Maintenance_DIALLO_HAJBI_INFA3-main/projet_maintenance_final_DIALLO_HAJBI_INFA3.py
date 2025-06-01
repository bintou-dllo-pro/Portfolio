import tkinter as tk
from tkinter import simpledialog, messagebox

class Shape:
    def draw(self, canvas):
        pass

class Circle(Shape):
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    def draw(self, canvas):
        canvas.create_oval(self.x - self.radius, self.y - self.radius, self.x + self.radius, self.y + self.radius, outline="black")

class Rectangle(Shape):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def draw(self, canvas):
        canvas.create_rectangle(self.x, self.y, self.x + self.width, self.y + self.height, outline="black")

class Bezier(Shape):
    def __init__(self, x1, y1, x2, y2, x3, y3, x4, y4):
        self.coords = (x1, y1, x2, y2, x3, y3, x4, y4)

    def draw(self, canvas):
        canvas.create_line(self.coords, smooth=True, splinesteps=100, fill="black")

class Node:
    def __init__(self, shape):
        self.shape = shape
        self.next = None

class Layer:
    def __init__(self):
        self.head = None

    def add_shape(self, shape):
        new_node = Node(shape)
        new_node.next = self.head
        self.head = new_node

    def remove_shape(self):
        if self.head is not None:
            self.head = self.head.next

    def render(self, canvas):
        current = self.head
        while current:
            current.shape.draw(canvas)
            current = current.next

class GestionLayer:
    def __init__(self):
        self.layers = []

    def add_layer(self):
        self.layers.append(Layer())

    def render(self, canvas):
        for layer in self.layers:
            layer.render(canvas)

class DessinVectoriel:
    def __init__(self, root):
        self.layer_manager = GestionLayer()
        self.root = root
        self.root.title("Application Dessin Vectoriel")
        self.canvas = tk.Canvas(self.root, width=600, height=400, bg="white")
        self.canvas.pack()
        self.menu()
        self.layer_manager.add_layer()

    def menu(self):
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)
        shape_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Shapes", menu=shape_menu)
        shape_menu.add_command(label="Add Circle", command=self.add_circle)
        shape_menu.add_command(label="Add Rectangle", command=self.add_rectangle)
        shape_menu.add_command(label="Add Bezier Curve", command=self.add_bezier)
        shape_menu.add_separator()
        shape_menu.add_command(label="Remove Shape", command=self.remove_shape)
        shape_menu.add_command(label="Add Layer", command=self.add_layer)
        shape_menu.add_command(label="Render", command=self.render)
        shape_menu.add_command(label="Exit", command=self.root.quit)

    def add_circle(self):
        x = self.get_int_input("Enter x coordinate:")
        y = self.get_int_input("Enter y coordinate:")
        radius = self.get_int_input("Enter radius:")
        if x is not None and y is not None and radius is not None:
            circle = Circle(x, y, radius)
            self.layer_manager.layers[-1].add_shape(circle)
            self.render()

    def add_rectangle(self):
        x = self.get_int_input("Enter x coordinate:")
        y = self.get_int_input("Enter y coordinate:")
        width = self.get_int_input("Enter width:")
        height = self.get_int_input("Enter height:")
        if x is not None and y is not None and width is not None and height is not None:
            rectangle = Rectangle(x, y, width, height)
            self.layer_manager.layers[-1].add_shape(rectangle)
            self.render()

    def add_bezier(self):
        x1 = self.get_int_input("Enter x1 coordinate:")
        y1 = self.get_int_input("Enter y1 coordinate:")
        x2 = self.get_int_input("Enter x2 coordinate:")
        y2 = self.get_int_input("Enter y2 coordinate:")
        x3 = self.get_int_input("Enter x3 coordinate:")
        y3 = self.get_int_input("Enter y3 coordinate:")
        x4 = self.get_int_input("Enter x4 coordinate:")
        y4 = self.get_int_input("Enter y4 coordinate:")
        if all(v is not None for v in [x1, y1, x2, y2, x3, y3, x4, y4]):
            bezier = Bezier(x1, y1, x2, y2, x3, y3, x4, y4)
            self.layer_manager.layers[-1].add_shape(bezier)
            self.render()

    def remove_shape(self):
        if self.layer_manager.layers:
            self.layer_manager.layers[-1].remove_shape()
            self.render()

    def add_layer(self):
        self.layer_manager.add_layer()
        self.render()

    def get_int_input(self, prompt):
        while True:
            value = simpledialog.askinteger("Input", prompt)
            if value is None: 
                return None
            return value

    def render(self):
        self.canvas.delete("all")
        self.layer_manager.render(self.canvas)

def main():
    root = tk.Tk()
    app = DessinVectoriel(root)
    root.mainloop()

if __name__ == "__main__":
    main()
