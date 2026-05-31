import torch

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
target_folder = project_root / "data"
target_folder.mkdir(parents=True, exist_ok=True)

from pytorch_lib.pytorch_utils import (
    prepare_mesh_pytorch,
    build_surface_F_pytorch,
    build_body_F_pytorch,
)
from utility.RENDER_PARAMS import (
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)

import time
from utility.mesh_generation import detect_right_boundary_faces_from_nodes
from utility.memory_monitor import GPUMemoryMonitor


import glfw
import ctypes
from OpenGL.GL import *

import numpy as np                   

from utility.render_util import (
    extract_surface_faces, 
    build_render_data_ref_cpu_32, 
    build_render_data_def_cpu_32, 

    CameraState, 
    mouse_button_callback, 
    cursor_position_callback, 
    scroll_callback, 
    key_callback,
    char_callback,

    perspective, 
    view_matrix,
    rotation_y,
    rotation_x,

    is_valid_number,

    create_program, 
)



def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()



WAIT_INPUT = 0
SHOW_RESULT = 1

def simulation_user_driven_via_CPU(
        nx, ny, nz, 
        L, W, H, tet,
        lam, mu,

        precompute_func,
        solver_func,

        precalculate_dofs_func,
        force_type,
        load,
        framework,
        version,
        
        rtol,
        maxiter,

        dtype,
        device,

        precompute_times,
        monitor_memory,
        inputs,
):
    
    torch.set_grad_enabled(False)

    # ----------------------------
    # Memory monitor
    # ----------------------------
    monitor = None
    if monitor_memory and device.type == "cuda":
        monitor = GPUMemoryMonitor(interval=0.05)
        monitor.start()
    

    data_records = {
        "framework": framework,
        "n_elems": 0,
        "version": version,
        "load": load,
        "precompute_runs": {},
        "interactive_runs": {},
        "peak_mem": None,
    }

    # print("=====================================================================")
    # print(f"Mesh: {(nx-1)*(ny-1)*(nz-1)*tet} elements, framework: {framework}, load:{load}, version: {version}")
    # print("=====================================================================")


    # ----------------------------
    # precompute
    # ----------------------------

    for precompute_run in range(1, precompute_times+1):
        sync(device)
        t0 = time.perf_counter()

        elements, nodes, elements_np, nodes_np = prepare_mesh_pytorch(
            nx, ny, nz, L, W, H, tet, dtype, device
        )
        nodes_np_32 = nodes_np.astype(np.float32)
        
        precomp = precompute_func(nodes, elements, lam, mu, dtype, device)

        n_nodes = nodes_np.shape[0]
        n_elem = elements_np.shape[0]
        data_records["n_elems"]= n_elem

        dofs_all, dofs_fix = precalculate_dofs_func(
            nodes, elements, device, W, H, ny, nz
        )

        sync(device)
        #print(f"Precomputation Time: {time.perf_counter() - t0:.4f} s")
        data_records["precompute_runs"][precompute_run] = time.perf_counter() - t0
    
    surfaces = extract_surface_faces(elements_np)

    # ----------------------------
    # Initialize OpenGL
    # ----------------------------
    if not glfw.init():
        raise Exception("GLFW init failed")
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # Required on macOS
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, "User-driven FEM", None, None)
    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)

    camera = CameraState()
    state = {
        "input": {"buffer": "", "enter": False},
        "camera": camera,
        "msg": "",
    }

    glfw.set_window_user_pointer(window, state)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    glfw.set_key_callback(window, key_callback)
    glfw.set_char_callback(window, char_callback)

    # ----------------------------
    # Initialize data
    # ----------------------------
    u_np_32 = np.zeros((n_nodes, 3), dtype=np.float32)

    vertices_ref_32 = build_render_data_ref_cpu_32(nodes_np_32, surfaces)
    vertices_def_32 = build_render_data_def_cpu_32(nodes_np_32, u_np_32, surfaces)

    VAO_ref = glGenVertexArrays(1)
    glBindVertexArray(VAO_ref)
    VBO_ref = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO_ref)
    glBufferData(GL_ARRAY_BUFFER, vertices_ref_32.nbytes, vertices_ref_32, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, None)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)

    VAO_def = glGenVertexArrays(1)
    glBindVertexArray(VAO_def)
    VBO_def = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO_def)
    glBufferData(GL_ARRAY_BUFFER, vertices_def_32.nbytes, vertices_def_32, GL_DYNAMIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, None)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)

    shader = create_program()
    mvp_loc = glGetUniformLocation(shader, "MVP")

    # ----------------------------
    # state + timing
    # ----------------------------
    state_mode = WAIT_INPUT
    first_frame_done_per_solver = False

    solver_cnt = 0
    time_records_per_run = None

    auto_idx = 0


    # ----------------------------
    # main loop
    # ----------------------------
    while not glfw.window_should_close(window):

        glfw.poll_events()

        input_buffer = state["input"]["buffer"]
        enter_pressed = state["input"]["enter"]

        # ---- common render setup ----
        glClearColor(0.9, 0.9, 0.9, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        model = rotation_y(camera.yaw) @ rotation_x(camera.pitch)
        proj = perspective(np.radians(45), WINDOW_WIDTH/WINDOW_HEIGHT, 0.1, 100.0)
        view = view_matrix(camera.zoom)
        MVP = proj @ view @ model

        glUseProgram(shader)
        glUniformMatrix4fv(mvp_loc, 1, GL_TRUE, MVP)

        # wait input
        if state_mode == WAIT_INPUT:

            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glBindVertexArray(VAO_ref)
            glDrawArrays(GL_TRIANGLES, 0, len(vertices_ref_32)//6)

            # draw deformed mesh too
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glBindVertexArray(VAO_def)
            glDrawArrays(GL_TRIANGLES, 0, len(vertices_def_32)//6)

            glfw.set_window_title(window, f"{n_elem} elements, {load}, {framework}, {version}, ::::::: Enter force: {input_buffer}")

            if inputs is not None and auto_idx < len(inputs):

                # simulate user input
                force_val = inputs[auto_idx]
                auto_idx += 1
                #print(f"----------- run {auto_idx} -----------", file=sys.stderr)

                enter_pressed = True
                input_buffer = str(force_val)

                #(optional) small delay for visualization
                #time.sleep(0.1)
            
            if enter_pressed:

                if not is_valid_number(input_buffer):
                    state["msg"] = f"Invalid input"
                    state["input"]["buffer"] = ""
                    state["input"]["enter"] = False
                    continue
                
                solver_cnt = solver_cnt + 1
                force_val = float(input_buffer)

                sync(device)
                t_start_per_run = time.perf_counter()
                first_frame_done_per_solver = False


                # -------- Build force --------
                if load == "compression":
                    force_vec = torch.tensor([force_val, 0.0, 0.0], dtype=dtype, device=device)
                elif load in ("gravity", "traction"):
                    force_vec = torch.tensor([0.0, 0.0, force_val], dtype=dtype, device=device)
                else:
                    raise ValueError("Unknown load")

                if force_type == "body":
                    F = build_body_F_pytorch(
                        n_nodes, dofs_all,
                        precomp["V_all"], force_vec,
                        dtype, device
                    )
                    F[dofs_fix] = 0.0
                else:
                    faces_np = detect_right_boundary_faces_from_nodes(elements_np, nodes_np, L)
                    faces_t = torch.tensor(faces_np, dtype=torch.long, device=device)
                    F = build_surface_F_pytorch(
                        nodes, faces_t,
                        force_vec, dtype, device
                    )
                    F[dofs_fix] = 0.0
                assert force_vec.dtype == dtype
                assert nodes.dtype == dtype

                # -------- SOLVE (blocking) --------
                u_t, cnt = solver_func(
                    precomp, dofs_all, dofs_fix,
                    lam, mu, F, n_elem,
                    rtol, maxiter,
                    load, framework, version,
                    plot_cvg=False,
                )
                assert u_t.dtype == dtype
                time_records_per_run = {}

                if u_t.requires_grad:
                    print("WARNING: autograd is ON", file=sys.stderr)

                sync(device)
                
                time_records_per_run["solve_stage_time"] = time.perf_counter() - t_start_per_run
                time_records_per_run["iterations"] = cnt
                time_records_per_run["force"] = force_val

                u_np_32[:] = u_t.detach().to(torch.float32).cpu().numpy().reshape(-1, 3)

                state_mode = SHOW_RESULT
                state["input"]["enter"] = False
                state["input"]["buffer"] = ""


        # show result
        elif state_mode == SHOW_RESULT:

            # reference
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glBindVertexArray(VAO_ref)
            glDrawArrays(GL_TRIANGLES, 0, len(vertices_ref_32)//6)

            # deformed
            vertices_def_32 = build_render_data_def_cpu_32(nodes_np_32, u_np_32, surfaces)
            glBindBuffer(GL_ARRAY_BUFFER, VBO_def)
            glBufferSubData(GL_ARRAY_BUFFER, 0, vertices_def_32.nbytes, vertices_def_32)

            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glBindVertexArray(VAO_def)
            glDrawArrays(GL_TRIANGLES, 0, len(vertices_def_32)//6)

            glfw.set_window_title(window, f"{n_elem} elements, {load}, {framework}, {version}, ::::::: Force: {force_val}, Iterations: {cnt}")

            # ---- first frame timing ----
            if not first_frame_done_per_solver:
                sync(device)
                glFinish()

                time_records_per_run["first_frame_latency"] =  time.perf_counter() - t_start_per_run
                first_frame_done_per_solver = True

                data_records["interactive_runs"][solver_cnt] = time_records_per_run
            


            # ---- Reset ----
            if inputs is not None and auto_idx >= len(inputs):
                glfw.set_window_should_close(window, True)

            elif inputs is not None and auto_idx < len(inputs) :
                # automatically go to next run
                u_np_32.fill(0.0)
                state_mode = WAIT_INPUT

                # clear input
                state["input"]["buffer"] = ""
                state["input"]["enter"] = False
                state["msg"] = ""

            else:
                if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS:
                    u_np_32.fill(0.0)

                    # rebuild undeformed geometry
                    vertices_def_32 = build_render_data_def_cpu_32(
                        nodes_np_32,
                        u_np_32,
                        surfaces
                    )

                    # upload to GPU
                    glBindBuffer(GL_ARRAY_BUFFER, VBO_def)

                    glBufferSubData(
                        GL_ARRAY_BUFFER,
                        0,
                        vertices_def_32.nbytes,
                        vertices_def_32
                    )

                    state_mode = WAIT_INPUT

                    # debounce
                    while glfw.get_key(window, glfw.KEY_R) == glfw.PRESS:
                        glfw.poll_events()

                    # Clear input state
                    state["input"]["buffer"] = ""
                    state["input"]["enter"] = False
                    state["msg"] = ""

        glfw.swap_buffers(window)

    glfw.destroy_window(window)
    glfw.terminate()


    # ----------------------------
    # Memory monitor stop
    # ----------------------------
    if monitor is not None:
        peak_mem = monitor.stop()
    else:
        peak_mem = 0
    data_records["peak_mem"]= peak_mem
    

    return data_records
