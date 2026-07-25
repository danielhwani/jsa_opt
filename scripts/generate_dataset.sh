#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ARGS=(
  --scene classroom/classroom.xml                 # scene file path
  --name classroom_overfit_jsa_512                # dataset name
  --out-data "$PROJECT_ROOT/data"                 # output data root

  --variant cuda_ad_rgb                           # GPU rendering

  --width 512                                     # output width
  --height 512                                    # output height

  --num-views 2                                   # number of overfit views

  --input-spp 4                                   # noisy RGB input spp
  --aov-spp 4                                     # AOV / feature spp

  --ref-spp 4096                                  # GT reference spp
  --ref-chunk-spp 512                              # chunk size for ETA/progress

  --camera-mode fixed                             # fixed base camera with jitter
  --base-origin -1.69049 1.27158 5.88653          # base camera origin (m03, m13, m23 of the camera matrix)
  --base-target -1.539196 1.2998871 4.898447      # base camera target (m02, m12, m22 + origin of the camera matrix) 
  --base-up -0.00428443 0.999599 0.027981         # Y-up camera up vector (m01, m11, m21 of the camera matrix)
  --base-fov 60

  --origin-jitter 0.00                            # camera origin perturbation
  --target-jitter 0.05                            # target perturbation

  --max-depth 17                                  # path max depth
  --rr-depth 5                                    # RR start depth

  --overwrite                                     # recreate dataset
  --write-npz                                     # also write NPZ immediately
  --seed 1234
)

python codes/generate_dataset.py "${ARGS[@]}"

# extract camera parameters from the XML 
# #!/usr/bin/env python3
# import argparse
# import xml.etree.ElementTree as ET
# import numpy as np


# def find_sensor(root, index=0):
#     sensors = root.findall(".//sensor")
#     if not sensors:
#         raise RuntimeError("No <sensor> found in XML.")
#     if index < 0 or index >= len(sensors):
#         raise RuntimeError(f"sensor index {index} out of range. Found {len(sensors)} sensors.")
#     return sensors[index]


# def get_named_child(parent, tag, name):
#     for child in parent.findall(tag):
#         if child.attrib.get("name") == name:
#             return child
#     return None


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("xml", help="Mitsuba XML scene file")
#     parser.add_argument("--sensor-index", type=int, default=0)
#     parser.add_argument("--target-distance", type=float, default=1.0)
#     parser.add_argument("--camera-mode", default="fixed")
#     args = parser.parse_args()

#     tree = ET.parse(args.xml)
#     root = tree.getroot()

#     sensor = find_sensor(root, args.sensor_index)

#     fov_node = get_named_child(sensor, "float", "fov")
#     if fov_node is None:
#         raise RuntimeError("No <float name='fov' ...> found under <sensor>.")
#     fov = float(fov_node.attrib["value"])

#     transform = get_named_child(sensor, "transform", "to_world")
#     if transform is None:
#         raise RuntimeError("No <transform name='to_world'> found under <sensor>.")

#     matrix_node = transform.find("matrix")
#     if matrix_node is None:
#         raise RuntimeError("No <matrix value='...'> found under sensor/to_world.")

#     vals = [float(v) for v in matrix_node.attrib["value"].split()]
#     if len(vals) != 16:
#         raise RuntimeError(f"Expected 16 matrix values, got {len(vals)}.")

#     M = np.array(vals, dtype=np.float64).reshape(4, 4)

#     origin = M[:3, 3]
#     up = M[:3, 1]
#     forward = M[:3, 2]
#     target = origin + args.target_distance * forward

#     def fmt(v):
#         return " ".join(f"{x:.10g}" for x in v)

#     print(f"--camera-mode {args.camera_mode}")
#     print(f"--base-origin {fmt(origin)}")
#     print(f"--base-target {fmt(target)}")
#     print(f"--base-up {fmt(up)}")
#     print(f"--base-fov {fov:.10g}")


# if __name__ == "__main__":
#     main()