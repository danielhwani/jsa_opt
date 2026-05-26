import mitsuba as mi
import pyexr
mi.set_variant('llvm_ad_rgb')
scene = mi.load_file("classroom/classroom.xml")
print("Scene loaded successfully.")
image = mi.render(scene,spp=16)
print("Rendered image shape:", image.shape)
mi.util.write_bitmap("my_first_render.exr", image)
