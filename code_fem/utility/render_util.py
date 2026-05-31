from curses import window
import glfw     # Window + input handling (creates OpenGL context)
from OpenGL.GL import *     # All OpenGL functions
import numpy as np
import re

from utility.RENDER_PARAMS import (
    WALL_POS,
    COEFFICIENT,
)



# Shader sources (GPU programs):
VERTEX_SHADER = """
    #version 330 core

    // buffer has [x, y, z,   r, g, b]
    // GPU reads: 
    // aPos   = (x, y, z)
    // aColor = (r, g, b)
    // layout, glsl keyword, layout(...),  extra instructions to the GPU about this variable
    // layout (location = 0) means this variable is bound to attribute slot 0
    // in means : this shader receives a 3D vector called aPos
    // togethter: Take data from attribute slot 0, and store it in aPos
    layout (location = 0) in vec3 aPos; //match glVertexAttribPointer(0, 3, ...) 
    layout (location = 1) in vec3 aColor; //match glVertexAttribPointer(1, 3, ...)

    // sends data to the fragment shader
    // pipeline: vertex shader -> rasterizer (interpolation) -> fragment shader
    // ourColor -> interpolated -> each pixel
    // out means "this variable will be OUTPUT from the vertex shader"
    out vec3 ourColor;

    // Model-View-Projection
    // glUniformMatrix4fv(mvp_loc, ..., MVP), meaning same value for ALL vertices
    uniform mat4 MVP;   


    void main()
    {
        // gl_Position, special built-in variable (fixed name, cannot rename)
        gl_Position = MVP * vec4(aPos, 1.0);  
        // aPos, aColor, ourColor are NOT special, they are user-defined names
        ourColor = aColor;  
    }
"""


# Fragment shader: runs once per pixel
# For each pixel:
#     set color = ourColor
FRAGMENT_SHADER = """
    #version 330 core
    in vec3 ourColor;   // receive from vertex shader
    out vec4 FragColor;

    void main()
    {   // 1.0 is alpha for transparency.
        FragColor = vec4(ourColor, 1.0); //FragColor, output pixel color, can rename
    }
"""


# Compile shader (CPU -> GPU)
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)  # Create empty shader object on GPU
    glShaderSource(shader, source) # Attach source code
    glCompileShader(shader) # Compile on GPU

    # Check compile errors
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compile error:\n{error}")

    return shader   # Return compiled shader


# Create shader program
# 1. compile vertex shader 2. compile fragment shader 3. link them together
def create_program():
    vertex = compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fragment = compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)

    program = glCreateProgram() # Create GPU program
    glAttachShader(program, vertex) # Attach vertex shader
    glAttachShader(program, fragment) # Attach fragment shader
    glLinkProgram(program) # Link into one program

    
    # check linking errors
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Program link error:\n{error}")

    # Shaders are no longer needed after linking
    glDeleteShader(vertex)
    glDeleteShader(fragment)
    
    # final GPU program
    return program  






def perspective(fov, aspect, near, far):
    f = 1.0 / np.tan(fov / 2.0)
    return np.array([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)


def view_matrix(zoom):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, -zoom],  # move camera back
        [0, 0, 0, 1]
    ], dtype=np.float32)


def rotation_y(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1]
    ], dtype=np.float32)


def rotation_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [ c, -s, 0, 0],
        [ s,  c, 0, 0],
        [ 0,  0, 1, 0],
        [ 0,  0, 0, 1]
    ], dtype=np.float32)


def rotation_x(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [1,  0,  0, 0],
        [0,  c, -s, 0],
        [0,  s,  c, 0],
        [0,  0,  0, 1]
    ], dtype=np.float32)




# arguments are required by GLFW callback
def mouse_button_callback(window, button, action, mods):
    
    camera = glfw.get_window_user_pointer(window)["camera"]

    if button == glfw.MOUSE_BUTTON_LEFT:
        camera.mouse_pressed = (action == glfw.PRESS)

        if action == glfw.PRESS:
            xpos, ypos = glfw.get_cursor_pos(window)
            camera.last_x = xpos
            camera.last_y = ypos


def cursor_position_callback(window, xpos, ypos):
    camera = glfw.get_window_user_pointer(window)["camera"]

    if camera.mouse_pressed:
        dx = xpos - camera.last_x
        dy = ypos - camera.last_y

        sensitivity = 0.005
        camera.yaw   += dx * sensitivity
        camera.pitch += dy * sensitivity

    camera.last_x = xpos
    camera.last_y = ypos



# convention in graphics:
# scroll up -> zoom in
# scroll down -> zoom out
def scroll_callback(window, xoffset, yoffset):
    camera = glfw.get_window_user_pointer(window)["camera"]
    #scroll up -> decrease zoom distance -> move closer
    #scroll down -> increase zoom distance -> move away
    camera.zoom -= yoffset * 0.2
    camera.zoom = max(0.6, min(2.0, camera.zoom))


def key_callback(window, key, scancode, action, mods):
    state = glfw.get_window_user_pointer(window)

    if action == glfw.PRESS:
        if key == glfw.KEY_ENTER:
            state["input"]["enter"] = True
        elif key == glfw.KEY_BACKSPACE:
            state["input"]["buffer"] = state["input"]["buffer"][:-1]
        elif key == glfw.KEY_Q:
            glfw.set_window_should_close(window, True)


def char_callback(window, codepoint):
    state = glfw.get_window_user_pointer(window)
    ch = chr(codepoint)
    # allow all printable characters
    if ch.isprintable():
        state["input"]["buffer"] += ch
        state["msg"] = ""   # clear error when typing



def is_valid_number(s):
    return re.match(r'^-?\d+(\.\d+)?$', s) is not None

class CameraState:
    def __init__(self):
        self.yaw = 0.0
        self.pitch = np.pi / 2
        self.zoom = 1.0
        self.last_x = 0.0
        self.last_y = 0.0
        self.mouse_pressed = False




# extract boundary triangle faces from tetrahedral mesh
def extract_surface_faces(elements_np):
    all_faces = np.vstack([
        elements_np[:, [0, 1, 2]],
        elements_np[:, [0, 1, 3]],
        elements_np[:, [0, 2, 3]],
        elements_np[:, [1, 2, 3]],
    ])

    # Sort vertex indices inside faces
    all_faces = np.sort(all_faces, axis=1)


    # Find unique faces + counts
    unique_faces, counts = np.unique(
        all_faces,
        axis=0,
        return_counts=True
    )

    # Surface faces appear exactly once
    surface_faces = unique_faces[counts == 1]

    return surface_faces.astype(np.int32)



def build_render_data_def_cpu_32(
    nodes_np,
    u_np,
    faces,
    wall_pos=WALL_POS,
    coefficient=COEFFICIENT,
):


    # geometry normalization
    min_coords = nodes_np.min(axis=0)
    max_coords = nodes_np.max(axis=0)

    center_y = (min_coords[1] + max_coords[1]) / 2
    center_z = (min_coords[2] + max_coords[2]) / 2
    anchor_x = min_coords[0]

    scale = np.max(max_coords - min_coords)


    # gather triangle vertices
    idx = faces.reshape(-1)

    pos = nodes_np[idx]    # (N,3)
    disp = u_np.reshape(-1,3)[idx]   # (N,3)


    # exact transform_point logic
    p_def = (pos + disp).copy()
    p_def[:,0] -= anchor_x
    p_def[:,1] -= center_y
    p_def[:,2] -= center_z

    p_def = p_def / scale * coefficient
    p_def[:,0] += wall_pos

    # deformation magnitude
    disp_mag_all = np.linalg.norm(u_np.reshape(-1,3), axis=1)
    disp_mag = disp_mag_all[idx]
    val_max = disp_mag_all.max()

    # colors
    color_val = disp_mag / (val_max + 1e-8)

    color_val = np.clip(color_val, 1e-8, 1.0)

    colors = np.stack([
        color_val,
        np.zeros_like(color_val),
        1.0 - color_val
    ], axis=1)


    # interleave position + color
    vertices = np.concatenate([p_def, colors], axis=1)
    return vertices.astype(np.float32).reshape(-1)



def build_render_data_ref_cpu_32(
    nodes_np,
    faces,
    wall_pos=WALL_POS,
    coefficient=COEFFICIENT,
):

    # geometry normalization
    min_coords = nodes_np.min(axis=0)
    max_coords = nodes_np.max(axis=0)

    center_y = (min_coords[1] + max_coords[1]) / 2
    center_z = (min_coords[2] + max_coords[2]) / 2
    anchor_x = min_coords[0]

    scale = np.max(max_coords - min_coords)

    # gather triangle vertices
    idx = faces.reshape(-1)
    pos = nodes_np[idx].copy()

    # exact transform_point logic
    pos[:,0] -= anchor_x
    pos[:,1] -= center_y
    pos[:,2] -= center_z
    pos = pos / scale * coefficient
    pos[:,0] += wall_pos

    # constant gray color
    colors = np.full((pos.shape[0], 3), 0.2, dtype=np.float32)

    # interleave position + color
    vertices = np.concatenate([pos, colors], axis=1)

    return vertices.astype(np.float32).reshape(-1)