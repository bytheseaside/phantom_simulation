"""
ParaView macro to probe active source at 10-20 EEG electrode locations.

This macro:
1. Creates probe points from CSV (TableToPoints) - shared across runs
2. Probes the ACTIVE source at all probe locations (nearest point lookup)
3. Saves results as both NPY (plain column) and CSV (with probe names)
4. Files are saved in the same directory as the source file

Usage:
1. Load source in ParaView
2. Select ONE source in Pipeline Browser
3. Run this macro (Macros menu or Tools > Python Shell > Run Script)
4. Repeat for each source
"""

from paraview.simple import *
import csv
import numpy as np
import os

# Configuration
PROBE_CSV_PATH = '/Users/brisarojas/Desktop/phantom_simulation/src/templates/template_probes/10_20_int_system_probes.csv'
ARRAY_NAME = 'u'  # Field to probe (voltage)
PROBE_TABLE_NAME = 'EEG Probe Points'


def get_or_create_probe_points():
    """Get existing probe points source or create new one from CSV.
    
    Returns:
        tuple: (tableToPoints proxy, list of probe names, list of coordinates)
    """
    # Read probe data from CSV (always needed for names/coords)
    probe_names = []
    probe_coords = []
    with open(PROBE_CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            probe_names.append(row['name'].strip())
            probe_coords.append((float(row['x']), float(row['y']), float(row['z'])))
    
    # Look for existing probe table by name
    existing = FindSource(PROBE_TABLE_NAME)
    if existing is not None:
        print(f"Using existing '{PROBE_TABLE_NAME}' from pipeline")
        return existing, probe_names, probe_coords
    
    # Create new TableToPoints from CSV
    print(f"Creating probe points from: {PROBE_CSV_PATH}")
    
    # Read CSV as table
    csvReader = CSVReader(FileName=[PROBE_CSV_PATH])
    csvReader.HaveHeaders = 1
    RenameSource('Probe CSV', csvReader)
    
    # Convert to points
    tableToPoints = TableToPoints(Input=csvReader)
    tableToPoints.XColumn = 'x'
    tableToPoints.YColumn = 'y'
    tableToPoints.ZColumn = 'z'
    tableToPoints.KeepAllDataArrays = 1  # Keep probe names
    RenameSource(PROBE_TABLE_NAME, tableToPoints)
    
    # Update to load data
    UpdatePipeline()
    
    # Configure visualization directly on TableToPoints
    display = Show(tableToPoints)
    display.SetRepresentationType('Point Gaussian')
    display.GaussianRadius = 0.003  # 3mm radius in meters
    display.ShaderPreset = 'Sphere'
    display.DiffuseColor = [0.0, 1.0, 0.0]  # Bright green
    
    print(f"✓ Created '{PROBE_TABLE_NAME}' with {len(probe_names)} electrodes (green spheres)")
    
    return tableToPoints, probe_names, probe_coords


def probe_source_at_locations(source, probe_names, probe_coords, array_name=ARRAY_NAME):
    """Probe a source at all electrode locations using nearest point lookup.
    
    Uses VTK's FindPoint to get the closest mesh point to each probe location.
    
    Args:
        source: ParaView source proxy (the data to probe)
        probe_names: List of probe names
        probe_coords: List of (x, y, z) tuples for each probe
        array_name: Name of the array to probe (default: 'u')
    
    Returns:
        tuple: (list of float values, list of actual coordinates where measured)
    """
    # Ensure pipeline is updated
    UpdatePipeline(proxy=source)
    
    # Fetch the mesh data from the source
    from paraview.servermanager import Fetch
    mesh_data = Fetch(source)
    
    # Get the array we want to probe
    point_data = mesh_data.GetPointData()
    array = point_data.GetArray(array_name)
    
    if array is None:
        print(f"  ERROR: Array '{array_name}' not found in source")
        print(f"  Available arrays: ", end="")
        for i in range(point_data.GetNumberOfArrays()):
            print(f"{point_data.GetArrayName(i)}, ", end="")
        print()
        return [float('nan')] * len(probe_names), []
    
    values = []
    actual_coords = []  # Store the actual mesh points used
    
    for i, name in enumerate(probe_names):
        # Get probe coordinates
        probe_coord = probe_coords[i]
        
        # Find the closest point in the mesh
        closest_point_id = mesh_data.FindPoint(probe_coord)
        
        if closest_point_id >= 0:
            # Get the value at the closest point
            value = array.GetValue(closest_point_id)
            
            # Get actual coordinates of the closest point
            closest_coord = mesh_data.GetPoint(closest_point_id)
            actual_coords.append(closest_coord)
            
            # Calculate distance for info
            dist = ((probe_coord[0] - closest_coord[0])**2 + 
                    (probe_coord[1] - closest_coord[1])**2 + 
                    (probe_coord[2] - closest_coord[2])**2)**0.5
            
            values.append(value)
            print(f"  {name}: {value:.6e} V  (snap: {dist*1000:.2f} mm)")
        else:
            print(f"  {name}: NaN (no point found)")
            values.append(float('nan'))
            actual_coords.append(probe_coord)  # Fallback to original
    
    return values, actual_coords


def create_actual_points_visualization(actual_coords, probe_names):
    """Create visualization for the actual mesh points where values were measured.
    
    Args:
        actual_coords: List of (x, y, z) tuples for actual measurement points
        probe_names: List of probe names
    """
    ACTUAL_POINTS_NAME = 'Actual Measurement Points'
    
    # Check if visualization already exists - delete and recreate
    existing = FindSource(ACTUAL_POINTS_NAME)
    if existing is not None:
        Delete(existing)
    
    # Create a temporary CSV with actual coordinates
    import tempfile
    temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    temp_csv.write('name,x,y,z\n')
    for name, coord in zip(probe_names, actual_coords):
        temp_csv.write(f'{name},{coord[0]},{coord[1]},{coord[2]}\n')
    temp_csv.close()
    
    # Read as CSV and convert to points
    csvReader = CSVReader(FileName=[temp_csv.name])
    csvReader.HaveHeaders = 1
    RenameSource('Actual Points CSV', csvReader)
    
    tableToPoints = TableToPoints(Input=csvReader)
    tableToPoints.XColumn = 'x'
    tableToPoints.YColumn = 'y'
    tableToPoints.ZColumn = 'z'
    tableToPoints.KeepAllDataArrays = 1
    RenameSource(ACTUAL_POINTS_NAME, tableToPoints)
    
    UpdatePipeline()
    
    # Configure visualization - magenta/pink color to contrast with green
    display = Show(tableToPoints)
    display.SetRepresentationType('Point Gaussian')
    display.GaussianRadius = 0.002  # 2mm radius (smaller than theoretical)
    display.ShaderPreset = 'Sphere'
    display.DiffuseColor = [1.0, 0.0, 1.0]  # Magenta
    
    # Clean up temp file
    os.unlink(temp_csv.name)
    
    print(f"✓ Created '{ACTUAL_POINTS_NAME}' (spheres)")
    
    return tableToPoints


def get_source_filename(source):
    """Extract filename from source proxy."""
    try:
        filename_prop = source.GetProperty('FileName')
        if filename_prop:
            filename = filename_prop.GetElement(0)
            return os.path.basename(filename)
    except:
        pass
    return str(source)


def get_source_directory(source):
    """Extract directory path from source proxy."""
    try:
        filename_prop = source.GetProperty('FileName')
        if filename_prop:
            filename = filename_prop.GetElement(0)
            return os.path.dirname(filename)
    except:
        pass
    return None


def save_probe_results(probe_names, values, base_filename, output_dir):
    """Save probe results in both NPY and CSV formats."""
    name_without_ext = os.path.splitext(base_filename)[0]
    
    values_array = np.array(values, dtype=np.float64)
    
    # Save NPY (plain column, no headers)
    npy_path = os.path.join(output_dir, f"{name_without_ext}.npy")
    np.save(npy_path, values_array)
    
    # Save CSV (with headers and probe names)
    csv_path = os.path.join(output_dir, f"{name_without_ext}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['PN', 'value [V]'])
        for name, value in zip(probe_names, values):
            writer.writerow([name, value])
    
    return npy_path, csv_path


def main():
    """Main execution function."""
    
    # Check if probe CSV exists
    if not os.path.exists(PROBE_CSV_PATH):
        print(f"ERROR: Probe CSV not found at: {PROBE_CSV_PATH}")
        return
    
    # Get active source
    source = GetActiveSource()
    if source is None:
        print("\nERROR: No source selected in Pipeline Browser")
        print("Please select a source and run the macro again")
        return
    
    # Get source information
    filename = get_source_filename(source)
    output_dir = get_source_directory(source)
    
    if output_dir is None:
        print(f"\nERROR: Could not determine output directory for source")
        print("Make sure you selected a file-based source (e.g., XDMF reader)")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    # Get or create probe points
    probePoints, probe_names, probe_coords = get_or_create_probe_points()
    
    # Probe at all locations
    print(f"\nProbing {len(probe_names)} electrode locations...")
    values, actual_coords = probe_source_at_locations(source, probe_names, probe_coords)
    
    # Create visualization of actual measurement points (magenta)
    create_actual_points_visualization(actual_coords, probe_names)
    
    # Check for NaN values
    nan_count = sum(1 for v in values if np.isnan(v))
    if nan_count > 0:
        print(f"\n⚠️  WARNING: {nan_count}/{len(values)} probes returned NaN")
    
    # Statistics
    valid_values = [v for v in values if not np.isnan(v)]
    if valid_values:
        print(f"\nStats: Min={min(valid_values):.4e}, Max={max(valid_values):.4e}, Mean={np.mean(valid_values):.4e} V")
    
    # Save results
    npy_path, csv_path = save_probe_results(probe_names, values, filename, output_dir)
    
    print(f"\n✓ Saved: {os.path.basename(npy_path)}")
    print(f"✓ Saved: {os.path.basename(csv_path)}")
    print(f"\nDone! Select next source and run again.")
    print(f"{'='*60}\n")


try:
    main()
except Exception as e:
    import traceback
    print("❌ Error:", e)
    traceback.print_exc()
