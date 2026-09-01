# /// script
# requires-python = ">=3.10, <3.13"
# dependencies = [
#     "pymol-open-source-whl",
# ]
# ///

import os
import sys

# Set environment variable for headless rendering
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import pymol
pymol.pymol_argv = ["pymol", "-cq"]
pymol.finish_launching()

from pymol import cmd

# Define paths
cif_path = "estruturas/6w8e.cif"
png_output = "estruturas/nmda_render.png"
pse_output = "estruturas/nmda_session.pse"

print(f"Loading structure from {cif_path}...")
cmd.load(cif_path, "nmda")

# Verify load
atom_count = cmd.count_atoms("all")
print(f"Loaded structure with {atom_count} atoms.")
if atom_count == 0:
    print("Error: No atoms loaded. Structure file might be empty or corrupted.")
    cmd.quit()
    sys.exit(1)

# Reset representations
cmd.hide("everything", "all")
cmd.show("cartoon", "nmda")

# Subunits of NMDA Receptor (PDB 6W8E is a GluN1-GluN2B heterotetramer)
# Chains A & B: GluN1
# Chains C & D: GluN2B
print("Coloring GluN1 and GluN2B subunits...")
cmd.color("skyblue", "chain A or chain B")
cmd.color("salmon", "chain C or chain D")

# Focus and orient
cmd.orient("nmda")

# Visual tweaks for premium look
cmd.bg_color("black")
cmd.set("ray_opaque_background", 1)
cmd.set("cartoon_fancy_helices", 1)
cmd.set("ambient", 0.4)
cmd.set("specular", 0.5)
cmd.set("shininess", 50)

# Save render
print(f"Rendering high-quality image to {png_output}...")
cmd.png(png_output, width=1200, height=900, dpi=150)

# Save session file
print(f"Saving PyMOL session to {pse_output}...")
cmd.save(pse_output)

# Quit PyMOL
print("Done. Quitting PyMOL.")
cmd.quit()
