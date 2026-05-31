import glfw
from OpenGL.GL import *

if not glfw.init():
    raise Exception("GLFW failed")

window = glfw.create_window(800, 600, "Test", None, None)
glfw.make_context_current(window)

print("OpenGL working!")

glfw.terminate()


