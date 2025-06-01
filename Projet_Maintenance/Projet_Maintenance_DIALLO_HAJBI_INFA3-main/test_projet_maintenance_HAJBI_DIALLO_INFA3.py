import unittest
from tkinter import Tk, Canvas
from projet_maintenance_final_DIALLO_HAJBI_INFA3 import Shape, Circle, Rectangle, Bezier, Node, Layer, GestionLayer, DessinVectoriel


class TestShapes(unittest.TestCase):

    def setUp(self):
        self.root = Tk()
        self.canvas = Canvas(self.root, width=600, height=400)
        self.canvas.pack()

    def tearDown(self):
        self.root.destroy()

    def test_circle_draw(self):
        circle = Circle(100, 100, 50)
        circle.draw(self.canvas)
        self.assertEqual(len(self.canvas.find_all()), 1, "Le cercle doit etre dessiné sur le canva")

    def test_rectangle_draw(self):
        rectangle = Rectangle(50, 50, 150, 100)
        rectangle.draw(self.canvas)
        self.assertEqual(len(self.canvas.find_all()), 1, "Le rectangle doit être dessiné sur le canva")

    def test_bezier_draw(self):
        bezier = Bezier(10, 10, 20, 20, 30, 30, 40, 40)
        bezier.draw(self.canvas)
        self.assertEqual(len(self.canvas.find_all()), 1, "Bezier doit être dessiné sur le canva")

    def test_layer_add_shape(self):
        layer = Layer()
        circle = Circle(100, 100, 50)
        layer.add_shape(circle)
        self.assertIsNotNone(layer.head, "La forme doit etre ajouté au calque")

    def test_layer_remove_shape(self):
        layer = Layer()
        circle = Circle(100, 100, 50)
        layer.add_shape(circle)
        layer.remove_shape()
        self.assertIsNone(layer.head, "La forme doit etre supprimée du calque")

    def test_layer_render(self):
        layer = Layer()
        circle = Circle(100, 100, 50)
        layer.add_shape(circle)
        layer.render(self.canvas)
        self.assertEqual(len(self.canvas.find_all()), 1, "Le calque doit afficher toutes les formes sur le canva")

    def test_gestion_layer_add_layer(self):
        manager = GestionLayer()
        manager.add_layer()
        self.assertEqual(len(manager.layers), 1, "Le calque doit etre ajoute au gestionnaire")

    def test_dessin_vectoriel_init(self):
        app = DessinVectoriel(self.root)
        self.assertIsInstance(app, DessinVectoriel, "L'application doit venirde DessinVectoriel")

if __name__ == '__main__':
    unittest.main()
