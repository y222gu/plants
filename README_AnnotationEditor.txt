========================================
  Plant Root Annotation Editor
  User Guide
========================================

1. DATA FOLDER STRUCTURE
------------------------
Prepare your data folder with an image/ subfolder. Two layouts are supported:

  Structured layout (training data):
    data_folder/
      image/
        Species/Microscope/Experiment/Sample/
          {Sample}_TRITC.tif
          {Sample}_FITC.tif
          {Sample}_DAPI.tif
      annotation/    (Required for "Correct GT" mode; auto-created on save)
        Species_Microscope_Experiment_Sample.txt
      prediction/    (Required for "Correct GT" and "Correct Predictions" modes)
        Species_Microscope_Experiment_Sample.txt

  Generic layout (new/external data):
    data_folder/
      image/
        sample_001/
          sample_001_TRITC.tif
          sample_001_FITC.tif
          sample_001_DAPI.tif
        sample_002/
          ...
      annotation/    (auto-created on save)
        sample_001.txt
      prediction/    (from predict.py)
        sample_001.txt

Each sample folder must contain three TIF files with _TRITC, _FITC, and
_DAPI suffixes. Annotation and prediction files are in YOLO polygon format.

2. EDITOR MODES
---------------
Select a mode from the "Mode" dropdown:

  Correct GT (3 panels)
    - View original image, edit ground truth, see predictions as reference
    - Requires: image/, annotation/, prediction/

  Correct Predictions (2 panels)
    - View original image, edit predictions
    - Requires: image/, prediction/
    - Saves corrected annotations to annotation/

  Create GT (2 panels)
    - View original image, draw annotations from scratch
    - Requires: image/
    - Saves new annotations to annotation/

3. CONTROLS
-----------
Navigation:
  A / Left Arrow      Previous sample
  D / Right Arrow     Next sample

Polygon Operations:
  N                   Start drawing new polygon (click to add points)
  E                   Enter vertex editing mode on selected polygon
  Enter               Confirm drawing or edits
  Escape              Cancel drawing or edits (reverts all changes)
  Delete / Backspace  Delete selected vertex (in edit mode) or polygon
  S                   Save annotations to file

Editing Vertices:
  - Drag vertices to move them
  - Hover over an edge to see a green "+" marker; click to add a vertex
  - Select a vertex and press Delete to remove it
  - Ctrl+Z undoes individual editing steps

Copy from Reference (Correct GT mode only):
  Ctrl+C              Copy selected reference polygon to editable panel
  C                   Copy ALL reference polygons to editable panel

Class Selection:
  1-4                 Set class for new polygon
                      1=Aerenchyma, 2=Outer Endodermis,
                      3=Inner Endodermis, 0=Whole Root

View:
  Mouse Wheel         Zoom in/out
  Middle/Right Drag   Pan the image
  H                   Reset zoom and center all panels (Home)
  Home button         Bottom-right corner, same as H

Undo/Redo:
  Ctrl+Z              Undo (removes last drawn point in draw mode)
  Ctrl+Shift+Z        Redo

4. VISIBILITY
-------------
Use the checkboxes in the "Visibility" section to show/hide annotation
classes. Hidden classes cannot be selected or edited.

5. SAVING
---------
Press S to save. Annotations are saved in YOLO polygon format to the
annotation/ subdirectory of your data folder (created automatically
if it does not exist). A prompt will appear if you navigate away or
close the editor with unsaved changes.