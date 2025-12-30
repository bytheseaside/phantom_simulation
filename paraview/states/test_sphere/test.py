# script-version: 2.0
# Catalyst state generated using paraview version 6.0.1
import paraview
paraview.compatibility.major = 6
paraview.compatibility.minor = 0

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# Create a new 'Bar Chart View'
barChartView1 = CreateView('XYBarChartView')
barChartView1.Set(
    ViewSize=[1152, 816],
    ChartTitle='Absolute Error Distribution',
    ShowLegend=0,
    LeftAxisTitle='Node count',
    LeftAxisUseCustomRange=1,
    LeftAxisRangeMinimum=-13980.40206298828,
    LeftAxisRangeMaximum=218262.79793701172,
    BottomAxisTitle='Error [V]',
    BottomAxisUseCustomRange=1,
    BottomAxisRangeMinimum=-0.0046759865659272305,
    BottomAxisRangeMaximum=0.08379761343407278,
    RightAxisUseCustomRange=1,
    RightAxisRangeMinimum=-0.35271359999999996,
    RightAxisRangeMaximum=7.012713600000001,
    TopAxisUseCustomRange=1,
    TopAxisRangeMinimum=-0.35271359999999996,
    TopAxisRangeMaximum=7.012713600000001,
)

# Create a new 'Line Chart View'
lineChartView1 = CreateView('XYChartView')
lineChartView1.Set(
    ViewSize=[1152, 816],
    LeftAxisUseCustomRange=1,
    LeftAxisRangeMinimum=1.218789346194826,
    LeftAxisRangeMaximum=2.2187893461948263,
    BottomAxisUseCustomRange=1,
    BottomAxisRangeMinimum=0.00024036815739236772,
    BottomAxisRangeMaximum=0.05024036815739237,
    RightAxisUseCustomRange=1,
    RightAxisRangeMaximum=6.66,
    TopAxisUseCustomRange=1,
    TopAxisRangeMaximum=6.66,
)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.Set(
    ViewSize=[1154, 816],
    AxesGrid='Grid Axes 3D Actor',
    CenterOfRotation=[7.3499977588653564e-06, 7.802620530128479e-06, 0.0],
    CameraPosition=[0.18728787217311776, 7.802620530128479e-06, 0.0],
    CameraFocalPoint=[7.3499977588653564e-06, 7.802620530128479e-06, 0.0],
    CameraViewUp=[0.0, 0.0, 1.0],
    CameraFocalDisk=1.0,
    CameraParallelScale=0.10390353501789373,
    OSPRayMaterialLibrary=materialLibrary1,
)

# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.Set(
    ViewSize=[1154, 816],
    AxesGrid='Grid Axes 3D Actor',
    CenterOfRotation=[-0.024989020079374313, 5.3104013204574585e-06, 0.0],
    CameraPosition=[0.2144411160685329, -0.0010895584499800472, 1.9275755779595322e-08],
    CameraFocalPoint=[-0.024989020079374313, 5.310401320457457e-06, 0.0],
    CameraViewUp=[8.050596537271097e-08, 3.5210893041446114e-05, 0.9999999993800934],
    CameraFocalDisk=1.0,
    CameraParallelScale=0.07498336980142574,
    OSPRayMaterialLibrary=materialLibrary1,
)

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'Xdmf3 Reader S'
sigma35int123ext222xdmf = Xdmf3ReaderS(registrationName='sigma=3,5-int=1,23-ext=2,22.xdmf', FileName=['/Users/brisarojas/Desktop/phantom_simulation/run_test_sphere/cases_sigma_3.5/sigma=3,5-int=1,23-ext=2,22.xdmf'])
sigma35int123ext222xdmf.Set(
    PointArrays=['u'],
    CellArrays=['Cell tags', 'Facet tags'],
)

# create a new 'Programmable Filter'
validation = ProgrammableFilter(registrationName='Validation', Input=sigma35int123ext222xdmf)
validation.Set(
    RequestInformationScript='',
    RequestUpdateExtentScript='',
    PythonPath='',
    Script="""import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

# PARAMETERS
r_int = 0.001
r_ext = 0.05
V1 = 1.23
V2 = 2.22

# Real VTK input
input_ds = self.GetInputDataObject(0, 0)

# Pass geometry + existing arrays through
output.CopyStructure(input_ds)
output.GetPointData().PassData(input_ds.GetPointData())
output.GetCellData().PassData(input_ds.GetCellData())

# Points -> Nx3 numpy
pts_vtk = input_ds.GetPoints().GetData()
pts = vtk_to_numpy(pts_vtk)              # shape (N,3)

# r = sqrt(x^2+y^2+z^2)
r = np.linalg.norm(pts, axis=1)

# u(r) = A/r + B with u(r_int)=V1, u(r_ext)=V2
A = (V2 - V1) / ((1.0 / r_ext) - (1.0 / r_int))
B = V2 - A / r_ext
u_ana = A / r + B

print("N points:", pts.shape[0])
print("x range:", pts[:,0].min(), pts[:,0].max())
print("y range:", pts[:,1].min(), pts[:,1].max())
print("z range:", pts[:,2].min(), pts[:,2].max())

print("r min/max:", r.min(), r.max())
print("r_int/r_ext:", r_int, r_ext)

print("A,B:", A, B)

print("u_ana min/max:", u_ana.min(), u_ana.max())
print("u_ana sample:", u_ana[:10])
u_num = vtk_to_numpy(input_ds.GetPointData().GetArray("u"))
print("u (numeric) min/max:", np.nanmin(u_num), np.nanmax(u_num))


# Attach as Point Data
vtk_u = numpy_to_vtk(u_ana, deep=True)
vtk_u.SetName("u_ana")
output.GetPointData().AddArray(vtk_u)
""",
)

# create a new 'Calculator'
relerr = Calculator(registrationName='Rel err', Input=validation)
relerr.Set(
    ResultArrayName='rel_err',
    Function='abs(u-u_ana)/abs(u_ana)',
)

# create a new 'Clip'
u_anasectionview = Clip(registrationName='u_ana section view', Input=validation)
u_anasectionview.ClipType = 'Plane'

# init the 'Plane' selected for 'HyperTreeGridClipper'
u_anasectionview.HyperTreeGridClipper.Origin = [7.3499977588653564e-06, 7.802620530128479e-06, 0.0]

# create a new 'Calculator'
abserr = Calculator(registrationName='Abs err', Input=validation)
abserr.Set(
    ResultArrayName='abs_err',
    Function='abs(u - u_ana)',
)

# create a new 'Descriptive Statistics'
descriptiveStatistics1 = DescriptiveStatistics(registrationName='DescriptiveStatistics1', Input=abserr,
    ModelInput=None)
descriptiveStatistics1.VariablesofInterest = ['abs_err']

# create a new 'Histogram'
histogram1 = Histogram(registrationName='Histogram1', Input=abserr)
histogram1.SelectInputArray = ['POINTS', 'abs_err']

# create a new 'Clip'
errabssectionview = Clip(registrationName='Err (abs) section view', Input=abserr)
errabssectionview.ClipType = 'Plane'

# init the 'Plane' selected for 'HyperTreeGridClipper'
errabssectionview.HyperTreeGridClipper.Origin = [7.3499977588653564e-06, 7.802620530128479e-06, 0.0]

# create a new 'Clip'
casesectionview = Clip(registrationName='Case section view', Input=sigma35int123ext222xdmf)
casesectionview.ClipType = 'Plane'

# init the 'Plane' selected for 'HyperTreeGridClipper'
casesectionview.HyperTreeGridClipper.Origin = [7.3499977588653564e-06, 7.802620530128479e-06, 0.0]

# create a new 'Descriptive Statistics'
descriptiveStatistics2 = DescriptiveStatistics(registrationName='DescriptiveStatistics2', Input=relerr,
    ModelInput=None)
descriptiveStatistics2.VariablesofInterest = ['rel_err']

# create a new 'Plot Over Line'
plot = PlotOverLine(registrationName='Plot', Input=validation)
plot.Set(
    Point1=[0.001, 0.0, 0.0],
    Point2=[0.05, 0.0, 0.0],
)

# create a new 'Clip'
errrelsectionview = Clip(registrationName='Err (rel) section view', Input=relerr)
errrelsectionview.ClipType = 'Plane'

# init the 'Plane' selected for 'HyperTreeGridClipper'
errrelsectionview.HyperTreeGridClipper.Origin = [7.3499977588653564e-06, 7.802620530128479e-06, 0.0]

# ----------------------------------------------------------------
# setup the visualization in view 'barChartView1'
# ----------------------------------------------------------------

# show data from histogram1
histogram1Display = Show(histogram1, barChartView1, 'XYBarChartRepresentation')

# trace defaults for the display properties.
histogram1Display.Set(
    AttributeType='Row Data',
    UseIndexForXAxis=0,
    XArrayName='bin_extents',
    SeriesVisibility=['bin_values'],
    SeriesLabel=['bin_extents', 'bin_extents', 'bin_values', 'Distribution of Absolute Error'],
    SeriesColor=['bin_extents', '0', '0', '0', 'bin_values', '0.8899977207183838', '0.10000763088464737', '0.11000228673219681'],
    SeriesOpacity=['bin_extents', '1', 'bin_values', '1'],
    SeriesPlotCorner=['bin_extents', '0', 'bin_values', '0'],
    SeriesLabelPrefix='',
)

# ----------------------------------------------------------------
# setup the visualization in view 'lineChartView1'
# ----------------------------------------------------------------

# show data from plot
plotDisplay = Show(plot, lineChartView1, 'XYChartRepresentation')

# trace defaults for the display properties.
plotDisplay.Set(
    UseIndexForXAxis=0,
    XArrayName='arc_length',
    SeriesVisibility=['u', 'u_ana'],
    SeriesLabel=['arc_length', 'arc_length', 'u', 'u', 'u_ana', 'u_ana', 'vtkValidPointMask', 'vtkValidPointMask', 'Points_X', 'Points_X', 'Points_Y', 'Points_Y', 'Points_Z', 'Points_Z', 'Points_Magnitude', 'Points_Magnitude'],
    SeriesColor=['arc_length', '0', '0', '0', 'u', '0.8899977111467154', '0.10000762951094835', '0.1100022888532845', 'u_ana', '0.220004577706569', '0.4899977111467155', '0.7199969481956207', 'vtkValidPointMask', '0.30000762951094834', '0.6899977111467155', '0.2899977111467155', 'Points_X', '0.6', '0.3100022888532845', '0.6399938963912413', 'Points_Y', '1', '0.5000076295109483', '0', 'Points_Z', '0.6500038147554742', '0.3400015259021897', '0.16000610360875867', 'Points_Magnitude', '0', '0', '0'],
    SeriesOpacity=['arc_length', '1', 'u', '1', 'u_ana', '1', 'vtkValidPointMask', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Points_Magnitude', '1'],
    SeriesPlotCorner=['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'arc_length', '0', 'u', '0', 'u_ana', '0', 'vtkValidPointMask', '0'],
    SeriesLabelPrefix='',
    SeriesLineStyle=['Points_Magnitude', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'arc_length', '1', 'u', '1', 'u_ana', '1', 'vtkValidPointMask', '1'],
    SeriesLineThickness=['Points_Magnitude', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'arc_length', '2', 'u', '2', 'u_ana', '2', 'vtkValidPointMask', '2'],
    SeriesMarkerStyle=['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'arc_length', '0', 'u', '0', 'u_ana', '0', 'vtkValidPointMask', '0'],
    SeriesMarkerSize=['Points_Magnitude', '4', 'Points_X', '4', 'Points_Y', '4', 'Points_Z', '4', 'arc_length', '4', 'u', '4', 'u_ana', '4', 'vtkValidPointMask', '4'],
)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from casesectionview
casesectionviewDisplay = Show(casesectionview, renderView1, 'UnstructuredGridRepresentation')

# get 2D transfer function for 'u'
uTF2D = GetTransferFunction2D('u')

# get color transfer function/color map for 'u'
uLUT = GetColorTransferFunction('u')
uLUT.Set(
    TransferFunction2D=uTF2D,
    RGBPoints=[
        # scalar, red, green, blue
        1.2300000190734863, 0.231373, 0.298039, 0.752941,
        1.725000023841858, 0.865003, 0.865003, 0.865003,
        2.2200000286102295, 0.705882, 0.0156863, 0.14902,
    ],
    ColorSpace='Diverging',
    NanColor=[1.0, 1.0, 0.0],
    ScalarRangeInitialized=1.0,
)

# trace defaults for the display properties.
casesectionviewDisplay.Set(
    Representation='Surface',
    ColorArrayName=['POINTS', 'u'],
    LookupTable=uLUT,
    DataAxesGrid='Grid Axes Representation',
    PolarAxes='Polar Axes Representation',
)

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
casesectionviewDisplay.ScaleTransferFunction.Points = [1.2300000190734863, 0.0, 0.5, 0.0, 2.2200000286102295, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
casesectionviewDisplay.OpacityTransferFunction.Points = [1.2300000190734863, 0.0, 0.5, 0.0, 2.2200000286102295, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for uLUT in view renderView1
uLUTColorBar = GetScalarBar(uLUT, renderView1)
uLUTColorBar.Set(
    Title='u',
    ComponentTitle='',
    DrawDataRange=1,
    DataRangeLabelFormat='.%2f',
)

# set color bar visibility
uLUTColorBar.Visibility = 1

# show color legend
casesectionviewDisplay.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView2'
# ----------------------------------------------------------------

# show data from errabssectionview
errabssectionviewDisplay = Show(errabssectionview, renderView2, 'UnstructuredGridRepresentation')

# get 2D transfer function for 'abs_err'
abs_errTF2D = GetTransferFunction2D('abs_err')

# get color transfer function/color map for 'abs_err'
abs_errLUT = GetColorTransferFunction('abs_err')
abs_errLUT.Set(
    TransferFunction2D=abs_errTF2D,
    RGBPoints=[
        # scalar, red, green, blue
        0.0, 0.231373, 0.298039, 0.752941,
        0.03827884768735823, 0.865003, 0.865003, 0.865003,
        0.07655769537471646, 0.705882, 0.0156863, 0.14902,
    ],
    ColorSpace='Diverging',
    NanColor=[1.0, 1.0, 0.0],
    ScalarRangeInitialized=1.0,
)

# trace defaults for the display properties.
errabssectionviewDisplay.Set(
    Representation='Surface',
    ColorArrayName=['POINTS', 'abs_err'],
    LookupTable=abs_errLUT,
    DataAxesGrid='Grid Axes Representation',
    PolarAxes='Polar Axes Representation',
)

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
errabssectionviewDisplay.ScaleTransferFunction.Points = [1.339550692591729e-10, 0.0, 0.5, 0.0, 0.05062432747490931, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
errabssectionviewDisplay.OpacityTransferFunction.Points = [1.339550692591729e-10, 0.0, 0.5, 0.0, 0.05062432747490931, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for abs_errLUT in view renderView2
abs_errLUTColorBar = GetScalarBar(abs_errLUT, renderView2)
abs_errLUTColorBar.Set(
    Title='abs_err',
    ComponentTitle='',
    DrawDataRange=1,
    DataRangeLabelFormat='.%2f',
)

# set color bar visibility
abs_errLUTColorBar.Visibility = 1

# show color legend
errabssectionviewDisplay.SetScalarBarVisibility(renderView2, True)

# ----------------------------------------------------------------
# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'abs_err'
abs_errPWF = GetOpacityTransferFunction('abs_err')
abs_errPWF.Set(
    Points=[0.0, 0.0, 0.5, 0.0, 0.07655769537471646, 1.0, 0.5, 0.0],
    ScalarRangeInitialized=1,
)

# get opacity transfer function/opacity map for 'u'
uPWF = GetOpacityTransferFunction('u')
uPWF.Set(
    Points=[1.2300000190734863, 0.0, 0.5, 0.0, 2.2200000286102295, 1.0, 0.5, 0.0],
    ScalarRangeInitialized=1,
)

# ----------------------------------------------------------------
# setup animation scene, tracks and keyframes
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# initialize the timekeeper
timeKeeper1.SuppressedTimeSources = sigma35int123ext222xdmf

# get time animation track
timeAnimationCue1 = GetTimeTrack()

# initialize the animation track

# get animation scene
animationScene1 = GetAnimationScene()

# initialize the animation scene
animationScene1.Set(
    ViewModules=[renderView1, lineChartView1, renderView2, barChartView1],
    Cues=timeAnimationCue1,
    AnimationTime=0.0,
    PlayMode='Snap To TimeSteps',
    NumberOfFrames=10,
)

# initialize the animation scene

# ------------------------------------------------------------------------------
# Catalyst options
from paraview import catalyst
options = catalyst.Options()

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from paraview.simple import SaveExtractsUsingCatalystOptions
    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)
