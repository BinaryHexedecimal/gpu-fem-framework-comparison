import glfw                         
from OpenGL.GL import *           
import numpy as np     
from .render_util import (
    extract_surface_faces, 
    build_render_data_ref_cpu_32, 
    build_render_data_def_cpu_32, 

    CameraState, 
    mouse_button_callback, 
    cursor_position_callback, 
    scroll_callback, 

    perspective, 
    view_matrix,
    rotation_y,
    rotation_x,

    create_program, 
)
from utility.RENDER_PARAMS import (
    WINDOW_HEIGHT,
    WINDOW_WIDTH
)



def render_openGL_static(nodes_np, elements_np, u_np):
    width = WINDOW_WIDTH
    height = WINDOW_HEIGHT

    faces = extract_surface_faces(elements_np)

    # initialize GLFW (must be first)
    if not glfw.init():
        raise Exception("GLFW init failed")

    # tell macOS which OpenGL version to use
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)

    # use modern OpenGL (no legacy functions)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    # Required on macOS
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    # create window + OpenGL context
    window = glfw.create_window(width, height, "Deformed beam (M1)", None, None)
    
    if not window:
        glfw.terminate()
        raise Exception("Failed to create window")
    

    camera = CameraState()
    state = {"camera": camera}
    glfw.set_window_user_pointer(window, state)
   
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_scroll_callback(window, scroll_callback)


    # create OpenGL context
    glfw.make_context_current(window)

    # let OpenGL compares depth (z-value) of pixels, and only draw the closest one. 
    glEnable(GL_DEPTH_TEST) 

    # vertex data (tetrahedron)
    vertices_ref_32 = build_render_data_ref_cpu_32(nodes_np, faces)
    vertices_def_32 = build_render_data_def_cpu_32(nodes_np, u_np, faces)

    # --- upload undeformed (wireframe) data ---
    # VAO tells which VBO is used and how attributes are laid out (position/color)
    VAO_ref = glGenVertexArrays(1) #create 1 VAO id
    glBindVertexArray(VAO_ref) #make it active

    VBO_ref = glGenBuffers(1) #create buffer id
    glBindBuffer(GL_ARRAY_BUFFER, VBO_ref) #bind it as vertex data buffer

    # GL_ARRAY_BUFFER means This buffer stores vertex attributes
    # vertices.nbytes is the size of data in bytes
    # vertices_ref_32 is the actual data
    # GL_STATIC_DRAW means This data will not change (static) and is used for drawing
    glBufferData(GL_ARRAY_BUFFER, vertices_ref_32.nbytes, vertices_ref_32, GL_STATIC_DRAW)


    # after the steps above, CPU (NumPy array) copied into GPU memory
    # After this, GPU owns the data, and we refer to it by the VBO id (VBO_ref)

    # position
    # how to read data from the VBO and feed into the vertex shader's input variables (aPos, aColor)
    glVertexAttribPointer(
        0,        # location, math the layout(location = 0) in vertex shader
        3,        # number of values, (x, y, z)
        GL_FLOAT, # type
        GL_FALSE, # normalize?
        6 * 4,    # stride (bytes), 6 floats per vertex, 4 bytes per float
        None      # offset, start at beginning
    )
    # tell GPU to use attribute slot 0
    glEnableVertexAttribArray(0)

    # color
    glVertexAttribPointer(
        1, # matches layout(location=1) -> aColor
        3, # (r, g, b)
        GL_FLOAT, 
        GL_FALSE, 
        6 * 4, 
        ctypes.c_void_p(3 * 4) # skip first 3 floats (x,y,z), start at r,g,b
    )
    glEnableVertexAttribArray(1)


    # after the block above, GPU knows that
    # for each vertex:
    #     read 6 floats
    #     location 0 -> first 3 floats -> aPos
    #     location 1 -> next 3 floats -> aColor
    # VBO = raw data
    # VAO = how to interpret data


    # --- upload deformed (filled) data ---
    VAO_def = glGenVertexArrays(1)
    glBindVertexArray(VAO_def)

    VBO_def = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO_def)   

    glBufferData(GL_ARRAY_BUFFER, vertices_def_32.nbytes, vertices_def_32, GL_STATIC_DRAW)

    # position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, None)
    glEnableVertexAttribArray(0)

    # color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)



    
    shader = create_program() # create shader program
    glUseProgram(shader) # activate the program

    # where MVP is stored inside shader program
    # "MVP" must match "uniform mat4 MVP;"
    mvp_loc = glGetUniformLocation(shader, "MVP")


    # Render loop

    while not glfw.window_should_close(window):

        glfw.poll_events()     # Handle input/events

        # Clear screen
        glClearColor(0.9, 0.9, 0.9, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


        model = rotation_y(camera.yaw) @ rotation_x(camera.pitch)
        aspect = width / height
        proj = perspective(np.radians(45), aspect, 0.1, 100.0)
        view = view_matrix(camera.zoom)
        MVP = proj @ view @ model


        # draw 3d objects
        glUseProgram(shader)
        MVP = proj @ view @ model
        # Send MVP into shader (uniform mat4 MVP), 
        # then vertex shader can set gl_Position = MVP * vec4(aPos, 1.0);
        glUniformMatrix4fv(mvp_loc, 1, GL_TRUE, MVP)

        # --- draw undeformed (wireframe) ---
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE) # draw only edges (no filled triangles)
        glBindVertexArray(VAO_ref) # use undeformed mesh data
        glDrawArrays(GL_TRIANGLES, 0, len(vertices_ref_32)//6)  # draw triangles using this data

        # --- draw deformed (filled) ---
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL) # draw solid triangles
        glBindVertexArray(VAO_def) # switch to deformed mesh
        glDrawArrays(GL_TRIANGLES, 0, len(vertices_def_32)//6)

        # show result on screen
        glfw.swap_buffers(window) 

    # Cleanup
    glfw.terminate()


