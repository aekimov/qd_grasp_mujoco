import mujoco
import numpy as np
import mediapy as media

xml = """
<mujoco>
    <option gravity="0 0 -10"/>
      <worldbody>
        <light name="top" pos="0 0 1"/>
        <body name="box_and_sphere" euler="0 0 -30">
          <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
          <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
          <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
        </body>
      </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

duration = 3.0
framerate = 60
frames = []

print(f"Recording {duration}s at {framerate} fps...")

mujoco.mj_resetData(model, data)

with mujoco.Renderer(model, height=480, width=640) as renderer:
    while data.time < duration:
        mujoco.mj_step(model, data)
        if len(frames) < data.time * framerate:
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)

print(f"Captured {len(frames)} frames")

video = np.array(frames)
media.write_video('test.mp4', video, fps=framerate)
print("Saved to test.mp4")